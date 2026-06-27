#include <c10/xpu/XPUGeneratorBridge.h>
#include <c10/util/Exception.h>

namespace c10::xpu {

static GetDefaultGeneratorFn g_get_gen_fn = nullptr;
static PhiloxStateFn g_philox_fn = nullptr;

void registerXPUGeneratorBridge(
    GetDefaultGeneratorFn get_gen,
    PhiloxStateFn philox) {
  g_get_gen_fn = get_gen;
  g_philox_fn = philox;
}

c10::intrusive_ptr<c10::GeneratorImpl> getDefaultXPUGeneratorBridge(
    int64_t device_index) {
  TORCH_CHECK(
      g_get_gen_fn != nullptr,
      "XPU generator bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling XPU generator functions.");
  return g_get_gen_fn(device_index);
}

std::pair<uint64_t, uint64_t> philoxXPUStateBridge(
    c10::GeneratorImpl* gen,
    uint64_t increment) {
  TORCH_CHECK(
      g_philox_fn != nullptr,
      "XPU generator bridge not registered. "
      "Ensure torch_xpu.dll is loaded before calling XPU generator functions.");
  return g_philox_fn(gen, increment);
}

} // namespace c10::xpu
