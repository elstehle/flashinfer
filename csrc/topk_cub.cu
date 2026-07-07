/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Prototype integration of CUB's DeviceBatchedTopK (NVIDIA/cccl PR #9224, thread-block-cluster
// segmented top-k) into FlashInfer, for A/B benchmarking against the radix/clusters top-k paths.
//
// The cluster backend requires SM90+ (thread-block clusters); the caller must gate to SM90+.
// Deterministic requests (gpu_to_gpu / tie-break preferences) also require SM90+. Defer the
// unsupported-arch diagnosis to the dispatch's runtime check so this TU still compiles for the JIT
// target set.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/device/device_batched_topk.cuh>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace {

// One concrete DeviceBatchedTopK::MaxPairs (arg-top-k) instantiation, explicit-temp-storage
// overload (graph-safe: the size query launches nothing, the run uses a caller-provided buffer).
//   - KeyT     : score dtype
//   - MaxSeg   : compile-time upper bound on per-segment (per-row) size
//   - MaxK     : compile-time upper bound on k
//   - DetReqT  : determinism requirement holder (not_guaranteed / gpu_to_gpu)
//   - TieReqT  : tie-break requirement holder (unspecified / prefer_smaller / prefer_larger)
// Segments live in a padded (num_rows, row_stride) buffer; each row selects top-k over its first
// `seg_size` elements (fixed == row width, or per-row `d_lengths`, int32). Values are
// segment-local indices, so `d_values_out` receives the arg-top-k indices.
template <typename KeyT, int MaxSeg, int MaxK, typename DetReqT, typename TieReqT>
cudaError_t cub_topk_maxpairs(const KeyT* d_in, cuda::std::int64_t row_stride,
                              cuda::std::int64_t num_rows, const cuda::std::int32_t* d_lengths,
                              cuda::std::int64_t fixed_len, cuda::std::int64_t k, KeyT* d_out_vals,
                              cuda::std::int32_t* d_out_idx, DetReqT det_req, TieReqT tie_req,
                              void* d_temp, size_t& temp_bytes, cudaStream_t stream) {
  if constexpr (MaxK > MaxSeg) {
    return cudaErrorInvalidValue;  // k > segment size can never be valid; skip instantiation
  } else {
    auto d_keys_in = cuda::make_strided_iterator(cuda::make_counting_iterator(d_in),
                                                 static_cast<cuda::std::ptrdiff_t>(row_stride));
    auto d_values_in =
        cuda::make_constant_iterator(cuda::make_counting_iterator(cuda::std::int32_t{0}));
    auto d_keys_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_out_vals),
                                                  static_cast<cuda::std::ptrdiff_t>(k));
    auto d_values_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_out_idx),
                                                    static_cast<cuda::std::ptrdiff_t>(k));

    auto k_param = cuda::args::immediate{k, cuda::args::bounds<1, MaxK>()};
    auto num_segments = cuda::args::immediate{num_rows};

    auto env = cuda::std::execution::env{
        cuda::execution::require(det_req, tie_req, cuda::execution::output_ordering::unsorted),
        cuda::stream_ref{stream}};

    if (d_lengths != nullptr) {
      auto segment_sizes = cuda::args::deferred_sequence{d_lengths, cuda::args::bounds<1, MaxSeg>()};
      return cub::DeviceBatchedTopK::MaxPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_values_in,
                                              d_values_out, segment_sizes, k_param, num_segments,
                                              env);
    }
    auto segment_sizes = cuda::args::immediate{fixed_len, cuda::args::bounds<1, MaxSeg>()};
    return cub::DeviceBatchedTopK::MaxPairs(d_temp, temp_bytes, d_keys_in, d_keys_out, d_values_in,
                                            d_values_out, segment_sizes, k_param, num_segments, env);
  }
}

// Map FlashInfer's (deterministic, tie_break) to CUB's acknowledged requirement pairs and dispatch.
//   tie_break: 0 = none, 1 = prefer smaller index, 2 = prefer larger index (1/2 imply deterministic)
// Non-deterministic  -> (not_guaranteed, unspecified)  [the fast filter path]
// Deterministic      -> (gpu_to_gpu,     unspecified)
// Tie-break small    -> (gpu_to_gpu,     prefer_smaller_index)
// Tie-break large    -> (gpu_to_gpu,     prefer_larger_index)
template <typename KeyT, int MaxSeg, int MaxK>
cudaError_t cub_topk_dispatch_req(const KeyT* d_in, cuda::std::int64_t row_stride,
                                  cuda::std::int64_t num_rows, const cuda::std::int32_t* d_lengths,
                                  cuda::std::int64_t fixed_len, cuda::std::int64_t k,
                                  KeyT* d_out_vals, cuda::std::int32_t* d_out_idx,
                                  bool deterministic, cuda::std::int64_t tie_break, void* d_temp,
                                  size_t& temp_bytes, cudaStream_t stream) {
  namespace det = cuda::execution::determinism;
  namespace tb = cuda::execution::tie_break;
#define FI_CUB_REQ(DET, TIE)                                                                   \
  cub_topk_maxpairs<KeyT, MaxSeg, MaxK>(d_in, row_stride, num_rows, d_lengths, fixed_len, k,   \
                                        d_out_vals, d_out_idx, (DET), (TIE), d_temp, temp_bytes, \
                                        stream)
  if (tie_break == 1) {
    return FI_CUB_REQ(det::gpu_to_gpu, tb::prefer_smaller_index);
  }
  if (tie_break == 2) {
    return FI_CUB_REQ(det::gpu_to_gpu, tb::prefer_larger_index);
  }
  if (deterministic) {
    return FI_CUB_REQ(det::gpu_to_gpu, tb::unspecified);
  }
  return FI_CUB_REQ(det::not_guaranteed, tb::unspecified);
#undef FI_CUB_REQ
}

// Single loose static ceiling instead of (max_len, k) buckets. Rationale (mirrors how FlashInfer's
// own radix/clusters top-k take k/seq_len at runtime and only specialize on dtype):
//   * MaxSeg: the cluster backend streams the segment in fixed-size chunks, so shared memory is
//     chunk-sized, not segment-sized; the dispatch uses runtime_max_segment_size under the ceiling.
//     A 1M ceiling covers all FlashInfer seq_lens with negligible cost.
//   * MaxK: only tunes the *baseline* sub-policy; the cluster backend (the SM90+ path used here)
//     is MaxK-independent, so this is just the upper bound k must satisfy.
// This keeps the instantiation count to dtypes x requirement-modes.
inline constexpr int kCubTopkMaxSeg = 1048576;  // 1M; CUB's supported segment-size ceiling
inline constexpr int kCubTopkMaxK = 4096;       // >= max runtime k on FlashInfer's grid

template <typename KeyT>
cudaError_t cub_topk_dispatch(const KeyT* d_in, cuda::std::int64_t row_stride,
                              cuda::std::int64_t num_rows, const cuda::std::int32_t* d_lengths,
                              cuda::std::int64_t max_len, cuda::std::int64_t k, KeyT* d_out_vals,
                              cuda::std::int32_t* d_out_idx, bool deterministic,
                              cuda::std::int64_t tie_break, void* d_temp, size_t& temp_bytes,
                              cudaStream_t stream) {
  if (max_len > kCubTopkMaxSeg || k > kCubTopkMaxK) {
    return cudaErrorInvalidValue;
  }
  return cub_topk_dispatch_req<KeyT, kCubTopkMaxSeg, kCubTopkMaxK>(
      d_in, row_stride, num_rows, d_lengths, max_len, k, d_out_vals, d_out_idx, deterministic,
      tie_break, d_temp, temp_bytes, stream);
}

void cub_topk_check(TensorView input, TensorView output_indices, TensorView output_values,
                    Optional<TensorView> maybe_lengths) {
  CHECK_INPUT(input);
  CHECK_INPUT(output_indices);
  CHECK_INPUT(output_values);
  CHECK_DIM(2, input);
  CHECK_DIM(2, output_indices);
  CHECK_DIM(2, output_values);
  const int64_t in_code = encode_dlpack_dtype(input.dtype());
  TVM_FFI_ICHECK(in_code == float32_code || in_code == float16_code || in_code == bfloat16_code)
      << "cub_topk supports float32, float16, or bfloat16";
  TVM_FFI_ICHECK(encode_dlpack_dtype(output_values.dtype()) == in_code)
      << "cub_topk output_values dtype must match input dtype";
  if (maybe_lengths.has_value()) {
    TVM_FFI_ICHECK(encode_dlpack_dtype(maybe_lengths.value().dtype()) == int32_code)
        << "cub_topk expects int32 lengths (matching FlashInfer's lengths dtype)";
  }
}

// Query the temp bytes for one dtype.
template <typename KeyT>
size_t cub_topk_query_impl(TensorView input, const cuda::std::int32_t* d_lengths,
                           cuda::std::int64_t k, bool deterministic, cuda::std::int64_t tie_break,
                           cudaStream_t stream) {
  size_t temp_bytes = 0;
  cudaError_t status = cub_topk_dispatch<KeyT>(
      static_cast<const KeyT*>(input.data_ptr()), input.stride(0), input.size(0), d_lengths,
      input.size(1), k, nullptr, nullptr, deterministic, tie_break, nullptr, temp_bytes, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk workspace-size query failed: " << cudaGetErrorString(status);
  return temp_bytes;
}

// Size query + run for one dtype, using a graph-safe workspace (or internal alloc when absent).
template <typename KeyT>
void cub_topk_exec_impl(TensorView input, TensorView output_indices, TensorView output_values,
                        const cuda::std::int32_t* d_lengths, cuda::std::int64_t k,
                        bool deterministic, cuda::std::int64_t tie_break,
                        Optional<TensorView> maybe_workspace, cudaStream_t stream) {
  const cuda::std::int64_t num_rows = input.size(0);
  const cuda::std::int64_t max_len = input.size(1);
  const cuda::std::int64_t row_stride = input.stride(0);
  const KeyT* d_in = static_cast<const KeyT*>(input.data_ptr());
  KeyT* d_out_vals = static_cast<KeyT*>(output_values.data_ptr());
  cuda::std::int32_t* d_out_idx = static_cast<cuda::std::int32_t*>(output_indices.data_ptr());

  size_t temp_bytes = 0;
  cudaError_t status =
      cub_topk_dispatch<KeyT>(d_in, row_stride, num_rows, d_lengths, max_len, k, d_out_vals,
                              d_out_idx, deterministic, tie_break, nullptr, temp_bytes, stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk size query failed: " << cudaGetErrorString(status);

  void* d_temp = nullptr;
  bool owned = false;
  if (maybe_workspace.has_value()) {
    d_temp = maybe_workspace.value().data_ptr();
    const size_t ws_bytes = static_cast<size_t>(maybe_workspace.value().size(0));
    TVM_FFI_ICHECK(ws_bytes >= temp_bytes)
        << "cub_topk workspace too small: need " << temp_bytes << " bytes, have " << ws_bytes;
  } else {
    TVM_FFI_ICHECK(cudaMallocAsync(&d_temp, temp_bytes, stream) == cudaSuccess)
        << "cub_topk temp alloc failed";
    owned = true;
  }

  status = cub_topk_dispatch<KeyT>(d_in, row_stride, num_rows, d_lengths, max_len, k, d_out_vals,
                                   d_out_idx, deterministic, tie_break, d_temp, temp_bytes, stream);
  if (owned) {
    cudaFreeAsync(d_temp, stream);
  }
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "cub_topk (DeviceBatchedTopK::MaxPairs) failed: " << cudaGetErrorString(status);
}

}  // namespace

// Query the temporary-storage bytes DeviceBatchedTopK needs for this problem + requirement, so the
// caller can allocate a graph-safe workspace once and reuse it across timed iterations.
int64_t cub_topk_workspace_size(TensorView input, Optional<TensorView> maybe_lengths, int64_t top_k,
                                bool deterministic, int64_t tie_break) {
  cub_topk_check(input, input, input, maybe_lengths);
  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());
  const cuda::std::int32_t* d_lengths =
      maybe_lengths.has_value()
          ? static_cast<const cuda::std::int32_t*>(maybe_lengths.value().data_ptr())
          : nullptr;
  const cuda::std::int64_t k = static_cast<cuda::std::int64_t>(top_k);
  const int64_t in_code = encode_dlpack_dtype(input.dtype());
  size_t temp_bytes = 0;
  if (in_code == float32_code) {
    temp_bytes = cub_topk_query_impl<float>(input, d_lengths, k, deterministic, tie_break, stream);
  } else if (in_code == float16_code) {
    temp_bytes = cub_topk_query_impl<half>(input, d_lengths, k, deterministic, tie_break, stream);
  } else {
    temp_bytes =
        cub_topk_query_impl<__nv_bfloat16>(input, d_lengths, k, deterministic, tie_break, stream);
  }
  return static_cast<int64_t>(temp_bytes);
}

// FFI entry: CUB-backed batched (segmented) top-k returning values + segment-local indices.
//   input          : (num_rows, max_len) scores, fp32 / fp16 / bf16
//   output_indices : (num_rows, k) int32  -- arg-top-k indices (segment-local)
//   output_values  : (num_rows, k), same dtype as input -- selected key values
//   maybe_lengths  : optional (num_rows,) int32 per-row valid segment sizes; absent => fixed = max_len
//   top_k          : k
//   deterministic  : if true, request gpu_to_gpu determinism (SM90+)
//   tie_break      : 0 none, 1 prefer smaller index, 2 prefer larger index (1/2 imply deterministic)
//   maybe_workspace: optional 1D uint8 temp-storage buffer (graph-safe). If absent, allocate/free
//                    internally via cudaMallocAsync (eager use only).
void cub_topk(TensorView input, TensorView output_indices, TensorView output_values,
              Optional<TensorView> maybe_lengths, int64_t top_k, bool deterministic,
              int64_t tie_break, Optional<TensorView> maybe_workspace) {
  cub_topk_check(input, output_indices, output_values, maybe_lengths);
  cudaSetDevice(input.device().device_id);
  auto stream = get_stream(input.device());
  const cuda::std::int32_t* d_lengths =
      maybe_lengths.has_value()
          ? static_cast<const cuda::std::int32_t*>(maybe_lengths.value().data_ptr())
          : nullptr;
  const cuda::std::int64_t k = static_cast<cuda::std::int64_t>(top_k);
  const int64_t in_code = encode_dlpack_dtype(input.dtype());
  if (in_code == float32_code) {
    cub_topk_exec_impl<float>(input, output_indices, output_values, d_lengths, k, deterministic,
                              tie_break, maybe_workspace, stream);
  } else if (in_code == float16_code) {
    cub_topk_exec_impl<half>(input, output_indices, output_values, d_lengths, k, deterministic,
                             tie_break, maybe_workspace, stream);
  } else {
    cub_topk_exec_impl<__nv_bfloat16>(input, output_indices, output_values, d_lengths, k,
                                      deterministic, tie_break, maybe_workspace, stream);
  }
}
