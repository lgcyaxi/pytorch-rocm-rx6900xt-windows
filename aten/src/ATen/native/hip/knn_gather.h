#pragma once

#include <cstdint>

namespace at {
class Tensor;
namespace native {

// ROCm: gather along dim=1 when the last index dim is a broadcast
// (GNN EdgeConv: feat.gather(1, idx.expand(..., C))). CUDA never calls these.
bool rocm_try_knn_gather(
    const Tensor& result,
    const Tensor& self,
    int64_t dim,
    const Tensor& index);

// Last-dim cat of two FP32 tensors (EdgeConv [center, neigh-center]).
bool rocm_try_cat2_last(const Tensor& out, const Tensor& a, const Tensor& b, int64_t dim);

} // namespace native
} // namespace at
