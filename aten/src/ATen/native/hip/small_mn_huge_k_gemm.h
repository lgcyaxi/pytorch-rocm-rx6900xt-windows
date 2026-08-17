#pragma once

#include <cstdint>

namespace at::native {

// rocBLAS Tensile fallback is slow when C is tiny and K is huge
// (nn.Linear / 1x1 conv grad_weight). ROCm-only; CUDA never calls this.
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
    int64_t ldc);

} // namespace at::native
