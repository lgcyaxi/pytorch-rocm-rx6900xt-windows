#include "hip/hip_runtime.h"

#include <ATen/hip/HIPContext.h>
#include <ATen/native/hip/small_mn_huge_k_gemm.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <c10/util/Exception.h>

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace at::native {
namespace {

constexpr int64_t kMaxDim = 128;
constexpr int64_t kMinK = 4096;
constexpr int kThreads = 256;
constexpr int kChunk = 64;
constexpr int kMaxOwned = 16;
constexpr int kMaxBlocks = 512;

__global__ void reduce_partials_kernel(
    int m,
    int n,
    int nblocks,
    const float* __restrict__ parts,
    float* __restrict__ C,
    int ldc) {
  const int nC = m * n;
  const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx >= nC) {
    return;
  }
  float s = 0.f;
  for (int b = 0; b < nblocks; ++b) {
    s += parts[b * nC + idx];
  }
  const int i = idx % m;
  const int j = idx / m;
  C[i + static_cast<int64_t>(j) * ldc] = s;
}

// Split-K into a workspace, then reduce. Avoids atomicAdd contention on tiny C.
__global__ void small_mn_huge_k_kernel(
    int m,
    int n,
    int k,
    bool ta,
    bool tb,
    const float* __restrict__ A,
    int lda,
    const float* __restrict__ B,
    int ldb,
    float* __restrict__ out,
    int out_stride_j,
    bool out_is_partial) {
  extern __shared__ float smem[];
  float* As = smem;
  float* Bs = smem + kChunk * m;

  const int tid = static_cast<int>(threadIdx.x);
  const int nC = m * n;
  float acc[kMaxOwned];
#pragma unroll
  for (int u = 0; u < kMaxOwned; ++u) {
    acc[u] = 0.f;
  }

  const int kStep = static_cast<int>(gridDim.x) * kChunk;
  for (int p0 = static_cast<int>(blockIdx.x) * kChunk; p0 < k; p0 += kStep) {
    const int ch = (p0 + kChunk <= k) ? kChunk : (k - p0);
    if (!ta) {
      for (int t = tid; t < ch * m; t += kThreads) {
        const int pk = t / m;
        const int i = t - pk * m;
        As[pk * m + i] = A[i + static_cast<int64_t>(p0 + pk) * lda];
      }
    } else {
      for (int t = tid; t < ch * m; t += kThreads) {
        const int i = t / ch;
        const int pk = t - i * ch;
        As[pk * m + i] = A[(p0 + pk) + static_cast<int64_t>(i) * lda];
      }
    }
    if (!tb) {
      for (int t = tid; t < ch * n; t += kThreads) {
        const int j = t / ch;
        const int pk = t - j * ch;
        Bs[pk * n + j] = B[(p0 + pk) + static_cast<int64_t>(j) * ldb];
      }
    } else {
      for (int t = tid; t < ch * n; t += kThreads) {
        const int pk = t / n;
        const int j = t - pk * n;
        Bs[pk * n + j] = B[j + static_cast<int64_t>(p0 + pk) * ldb];
      }
    }
    __syncthreads();
    for (int pk = 0; pk < ch; ++pk) {
      int u = 0;
      for (int idx = tid; idx < nC; idx += kThreads, ++u) {
        const int i = idx % m;
        const int j = idx / m;
        acc[u] += As[pk * m + i] * Bs[pk * n + j];
      }
    }
    __syncthreads();
  }

  int u = 0;
  for (int idx = tid; idx < nC; idx += kThreads, ++u) {
    if (out_is_partial) {
      out[static_cast<int64_t>(blockIdx.x) * nC + idx] = acc[u];
    } else {
      const int i = idx % m;
      const int j = idx / m;
      out[i + static_cast<int64_t>(j) * out_stride_j] = acc[u];
    }
  }
}

} // namespace

bool rocm_small_mn_huge_k_sgemm(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    float alpha,
    const float* a,
    int64_t lda,
    const float* b,
    int64_t ldb,
    float beta,
    float* c,
    int64_t ldc) {
  if (m <= 0 || n <= 0 || k < kMinK) {
    return false;
  }
  if (m > kMaxDim || n > kMaxDim) {
    return false;
  }
  // ~40 GB/s. Wins only for tiny C (EdgeConv / embed dW). 64x64 and 192x64
  // dW are faster on rocBLAS / a pinned Tensile ID.
  if (m * n > 512) {
    return false;
  }
  if (alpha != 1.0f || beta != 0.0f) {
    return false;
  }
  static const bool disabled = []() {
    const char* env = std::getenv("PYTORCH_ROCM_SMALL_MN_GEMM");
    return env != nullptr && std::strcmp(env, "0") == 0;
  }();
  if (disabled) {
    return false;
  }
  if (a == nullptr || b == nullptr || c == nullptr) {
    return false;
  }
  if (lda < 1 || ldb < 1 || ldc < m) {
    return false;
  }
  const int64_t int_max = std::numeric_limits<int>::max();
  if (m > int_max || n > int_max || k > int_max || lda > int_max ||
      ldb > int_max || ldc > int_max) {
    return false;
  }

  const bool ta = std::tolower(static_cast<unsigned char>(transa)) != 'n';
  const bool tb = std::tolower(static_cast<unsigned char>(transb)) != 'n';

  const int mi = static_cast<int>(m);
  const int ni = static_cast<int>(n);
  const int ki = static_cast<int>(k);
  const int nC = mi * ni;
  const int ntiles = (ki + kChunk - 1) / kChunk;
  const int nblocks = ntiles < kMaxBlocks ? ntiles : kMaxBlocks;
  const size_t smem = sizeof(float) * static_cast<size_t>(kChunk) *
      static_cast<size_t>(mi + ni);
  hipStream_t stream = at::cuda::getCurrentCUDAStream();

  if (nblocks == 1) {
    small_mn_huge_k_kernel<<<1, kThreads, smem, stream>>>(
        mi,
        ni,
        ki,
        ta,
        tb,
        a,
        static_cast<int>(lda),
        b,
        static_cast<int>(ldb),
        c,
        static_cast<int>(ldc),
        false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
  }

  auto& allocator = *c10::cuda::CUDACachingAllocator::get();
  auto parts_buf = allocator.allocate(
      sizeof(float) * static_cast<size_t>(nblocks) * static_cast<size_t>(nC));
  float* parts = static_cast<float*>(parts_buf.get());

  small_mn_huge_k_kernel<<<nblocks, kThreads, smem, stream>>>(
      mi,
      ni,
      ki,
      ta,
      tb,
      a,
      static_cast<int>(lda),
      b,
      static_cast<int>(ldb),
      parts,
      nC,
      true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int zero_blocks = (nC + kThreads - 1) / kThreads;
  reduce_partials_kernel<<<zero_blocks, kThreads, 0, stream>>>(
      mi, ni, nblocks, parts, c, static_cast<int>(ldc));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return true;
}

} // namespace at::native
