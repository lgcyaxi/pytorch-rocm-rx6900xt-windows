#pragma once

#include <c10/core/GeneratorImpl.h>
#include <c10/xpu/XPUMacros.h>
#include <utility>

namespace c10::xpu {

// Function pointer types for generator operations whose implementations
// reside in torch_xpu.dll.  Kernel DLLs (built separately in
// BUILD_SEPARATE_OPS mode) call these through c10_xpu.dll to avoid a
// link-time dependency on torch_xpu.dll.

using GetDefaultGeneratorFn = c10::intrusive_ptr<c10::GeneratorImpl> (*)(
    int64_t device_index);
using PhiloxStateFn = std::pair<uint64_t, uint64_t> (*)(
    c10::GeneratorImpl* gen,
    uint64_t increment);

// Register the bridge implementations.  Called once by torch_xpu during
// XPU generator initialization.
C10_XPU_API void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox);

// Bridge accessors for kernel DLLs.  These forward to the function
// pointers registered by torch_xpu, or raise an error if not yet
// registered.

C10_XPU_API c10::intrusive_ptr<c10::GeneratorImpl> getDefaultXPUGeneratorBridge(
    int64_t device_index);

C10_XPU_API std::pair<uint64_t, uint64_t> philoxXPUStateBridge(
    c10::GeneratorImpl* gen,
    uint64_t increment);

} // namespace c10::xpu
