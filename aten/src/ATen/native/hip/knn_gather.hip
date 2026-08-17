#include "hip/hip_runtime.h"

#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/native/hip/knn_gather.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace at::native {
namespace {

constexpr int kThreads = 256;
constexpr int64_t kMaxSmemBytes = 48 * 1024;
constexpr int64_t kMaxNeighbors = 4096;

bool gnn_layout_disabled() {
  const char* env = std::getenv("PYTORCH_ROCM_GNN_LAYOUT");
  if (env != nullptr && std::strcmp(env, "0") == 0) {
    return true;
  }
  env = std::getenv("PYTORCH_ROCM_KNN_GATHER");
  return env != nullptr && std::strcmp(env, "0") == 0;
}

bool knn_layout_ok(
    const Tensor& feat,
    const Tensor& index,
    const Tensor& other,
    int64_t dim,
    int64_t* M,
    int64_t* P,
    int64_t* C,
    int64_t* N) {
  if (dim != 1 || feat.dim() != 3 || index.dim() != 3 || other.dim() != 3) {
    return false;
  }
  if (feat.scalar_type() != kFloat || other.scalar_type() != kFloat) {
    return false;
  }
  if (index.scalar_type() != kLong && index.scalar_type() != kInt) {
    return false;
  }
  if (!feat.is_contiguous() || !other.is_contiguous()) {
    return false;
  }
  if (index.stride(2) != 0 || index.stride(1) != 1) {
    return false;
  }
  *M = feat.size(0);
  *P = feat.size(1);
  *C = feat.size(2);
  *N = index.size(1);
  if (*M <= 0 || *P <= 0 || *C <= 0 || *N <= 0 || *N > kMaxNeighbors) {
    return false;
  }
  if (index.size(0) != *M || index.size(2) != *C) {
    return false;
  }
  if (other.size(0) != *M || other.size(1) != *N || other.size(2) != *C) {
    return false;
  }
  if (*P * *C * static_cast<int64_t>(sizeof(float)) > kMaxSmemBytes) {
    return false;
  }
  const int64_t int_max = std::numeric_limits<int>::max();
  if (*M > int_max || *P > int_max || *C > int_max || *N > int_max) {
    return false;
  }
  if (index.stride(0) > int_max || index.stride(0) < 1) {
    return false;
  }
  return true;
}

template <typename index_t>
__global__ void knn_gather_fwd_kernel(
    const float* __restrict__ feat,
    const index_t* __restrict__ idx,
    float* __restrict__ out,
    int P,
    int C,
    int N,
    int idx_batch_stride) {
  extern __shared__ float smem[];
  const int m = static_cast<int>(blockIdx.x);
  const float* f = feat + static_cast<size_t>(m) * static_cast<size_t>(P) * C;
  for (int i = static_cast<int>(threadIdx.x); i < P * C;
       i += static_cast<int>(blockDim.x)) {
    smem[i] = f[i];
  }
  __syncthreads();

  float* o = out + static_cast<size_t>(m) * static_cast<size_t>(N) * C;
  const index_t* ix = idx + static_cast<size_t>(m) * idx_batch_stride;
  const int nC = N * C;
  for (int t = static_cast<int>(threadIdx.x); t < nC;
       t += static_cast<int>(blockDim.x)) {
    const int n = t / C;
    const int c = t - n * C;
    const int j = static_cast<int>(ix[n]);
    o[t] = smem[j * C + c];
  }
}

// Last-dim cat of two FP32 rows. k_repeat>1: src0 repeats every k rows
// (broadcast over the k-neighbor dim).
__global__ void cat2_last_kernel(
    const float* __restrict__ src0,
    const float* __restrict__ src1,
    float* __restrict__ out,
    int64_t rows,
    int C,
    int k_repeat) {
  const int vec = C >> 2;
  const int64_t nwork = rows * vec;
  for (int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       t < nwork;
       t += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = t / vec;
    const int v = static_cast<int>(t - row * vec);
    const int64_t row0 = k_repeat > 1 ? row / k_repeat : row;
    const float4 a =
        reinterpret_cast<const float4*>(src0 + row0 * C)[v];
    const float4 b =
        reinterpret_cast<const float4*>(src1 + row * C)[v];
    float4* dst = reinterpret_cast<float4*>(out + row * (static_cast<int64_t>(C) << 1));
    dst[v] = a;
    dst[v + vec] = b;
  }
}

} // namespace

bool rocm_try_knn_gather(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  if (gnn_layout_disabled()) {
    return false;
  }
  int64_t M = 0, P = 0, C = 0, N = 0;
  if (!knn_layout_ok(self, index, result, dim, &M, &P, &C, &N)) {
    return false;
  }
  if (C < 32) {
    return false;
  }
  const int mi = static_cast<int>(M);
  const int pi = static_cast<int>(P);
  const int ci = static_cast<int>(C);
  const int ni = static_cast<int>(N);
  const int idx_batch_stride = static_cast<int>(index.stride(0));
  const size_t smem =
      sizeof(float) * static_cast<size_t>(pi) * static_cast<size_t>(ci);
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "rocm_knn_gather", [&]() {
    knn_gather_fwd_kernel<index_t>
        <<<mi, kThreads, smem, stream>>>(
            self.const_data_ptr<float>(),
            index.const_data_ptr<index_t>(),
            result.mutable_data_ptr<float>(),
            pi,
            ci,
            ni,
            idx_batch_stride);
  });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

bool rocm_try_cat2_last(
    const Tensor& out,
    const Tensor& a,
    const Tensor& b,
    int64_t dim) {
  if (gnn_layout_disabled()) {
    return false;
  }
  if (a.scalar_type() != kFloat || b.scalar_type() != kFloat ||
      out.scalar_type() != kFloat) {
    return false;
  }
  if (a.dim() != b.dim() || a.dim() != out.dim() || a.dim() < 2) {
    return false;
  }
  dim = dim < 0 ? dim + a.dim() : dim;
  if (dim != a.dim() - 1) {
    return false;
  }
  if (!out.is_contiguous() || !b.is_contiguous()) {
    return false;
  }
  const int64_t C = a.size(-1);
  if (C <= 0 || (C & 3) != 0 || b.size(-1) != C || out.size(-1) != 2 * C) {
    return false;
  }
  for (int64_t i = 0; i < dim; ++i) {
    if (a.size(i) != b.size(i) || out.size(i) != a.size(i)) {
      return false;
    }
  }
  int k_repeat = 1;
  const float* src0 = a.const_data_ptr<float>();
  if (!a.is_contiguous()) {
    if (a.stride(-1) != 1 || a.stride(-2) != 0) {
      return false;
    }
    k_repeat = static_cast<int>(a.size(-2));
    if (k_repeat < 1) {
      return false;
    }
    if (a.dim() >= 3 && a.stride(-3) != C) {
      return false;
    }
    int64_t expected = C;
    if (a.dim() >= 3) {
      expected = C * a.size(-3);
    }
    for (int64_t i = a.dim() - 4; i >= 0; --i) {
      if (a.stride(i) != expected) {
        return false;
      }
      expected *= a.size(i);
    }
  }
  const int64_t rows = b.numel() / C;
  if (rows <= 0) {
    return false;
  }
  const int64_t nwork = rows * (C >> 2);
  int nblocks = static_cast<int>((nwork + kThreads - 1) / kThreads);
  if (nblocks > 2048) {
    nblocks = 2048;
  }
  if (nblocks < 1) {
    nblocks = 1;
  }
  hipStream_t stream = at::cuda::getCurrentCUDAStream();
  cat2_last_kernel<<<nblocks, kThreads, 0, stream>>>(
      src0,
      b.const_data_ptr<float>(),
      out.mutable_data_ptr<float>(),
      rows,
      static_cast<int>(C),
      k_repeat);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

} // namespace at::native
