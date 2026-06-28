#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/Dispatch_v2.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/cuda/Loops.cuh>


// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native { namespace {

enum class EqOpType {EQ, NE};

} // namespace (anonymous)

template <typename scalar_t>
void compare_eq_ne_scalar_kernel(TensorIteratorBase& iter, EqOpType op, scalar_t rhs) {
  if (op == EqOpType::EQ) {
    gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t lhs) -> bool {
      return lhs == rhs;
    });
  } else {
    gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t lhs) -> bool {
      return lhs != rhs;
    });
  }
}

template <typename scalar_t>
void compare_eq_ne_kernel_impl(TensorIteratorBase& iter, EqOpType op) {
  // If either input is a CPU scalar, compare with the scalar on the right.
  // Equality and inequality are symmetric, so this preserves semantics.
  if (iter.is_cpu_scalar(1)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(1);
    iter.remove_operand(1);
    const DeviceGuard device_guard(iter.device(1));
    compare_eq_ne_scalar_kernel(iter, op, rhs);
  } else if (iter.is_cpu_scalar(2)) {
    const scalar_t rhs = iter.scalar_value<scalar_t>(2);
    iter.remove_operand(2);
    compare_eq_ne_scalar_kernel(iter, op, rhs);
  } else if (op == EqOpType::EQ) {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a == b;
    });
  } else {
    gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> bool {
      return a != b;
    });
  }
}

namespace {

C10_NOINLINE void compare_eq_ne_kernel(TensorIteratorBase &iter, EqOpType op) {
  AT_DISPATCH_V2(iter.common_dtype(), "compare_eq_ne_cuda", AT_WRAP([&]() {
    compare_eq_ne_kernel_impl<scalar_t>(iter, op);
  }), AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX), kComplexHalf, kBComplex32, kHalf, kBFloat16, kBool, AT_EXPAND(AT_FLOAT8_TYPES), AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES), kFloat4_e2m1fn_x2);
}

void eq_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::EQ);
}

void ne_kernel_cuda(TensorIteratorBase& iter) {
  compare_eq_ne_kernel(iter, EqOpType::NE);
}

REGISTER_DISPATCH(eq_stub, &eq_kernel_cuda)
REGISTER_DISPATCH(ne_stub, &ne_kernel_cuda)

} // namespace
} // namespace at::native
