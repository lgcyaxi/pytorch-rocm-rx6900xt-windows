#include <torch/csrc/distributed/c10d/watchdog/init.hpp>

#ifdef TORCH_USE_LIBUV

#include <ATen/DeviceAccelerator.h>
#include <c10/util/Exception.h>
#include <pybind11/chrono.h>
#include <pybind11/functional.h>
#include <torch/csrc/distributed/c10d/watchdog/Watchdog.hpp>
#include <torch/csrc/utils/pybind.h>

namespace py = pybind11;

namespace c10d::watchdog {
namespace {

// Wraps a Python-provided callback so that an exception raised on the watchdog
// loop thread is reported via sys.unraisablehook instead of escaping into the
// loop. pybind already makes the underlying std::function GIL-safe to call and
// destroy.
Callback wrapCallback(std::function<void()> fn) {
  return [fn = std::move(fn)]() {
    try {
      fn();
    } catch (py::error_already_set& e) {
      py::gil_scoped_acquire gil;
      e.discard_as_unraisable("torch.distributed._watchdog callback");
    } catch (const std::exception& e) {
      TORCH_WARN("torch.distributed._watchdog callback raised: ", e.what());
    }
  };
}

c10::Stream currentStream() {
  return at::accelerator::getCurrentStream(at::accelerator::getDeviceIndex());
}

} // namespace

void initWatchdogBindings(py::module& module) {
  auto m = module.def_submodule(
      "_distributed_c10d_watchdog", "c10d watchdog bindings");

  py::class_<Handle>(m, "_WatchdogHandle").def("cancel", &Handle::cancel);

  py::class_<Watchdog, std::shared_ptr<Watchdog>>(m, "_Watchdog")
      // Destruction joins the watchdog loop thread, which may be acquiring the
      // GIL to run or release a Python callback; release the GIL while it
      // happens to avoid a deadlock.
      .def(
          py::init([](std::chrono::milliseconds pollInterval) {
            return std::shared_ptr<Watchdog>(
                makeWatchdog(pollInterval).release(), [](Watchdog* p) {
                  py::gil_scoped_release release;
                  delete p;
                });
          }),
          py::arg("poll_interval") = std::chrono::seconds(1))
      .def_static("_singleton", &Watchdog::singleton)
      .def(
          "cpu_timeout",
          [](const std::shared_ptr<Watchdog>& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self->cpu_timeout(
                timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def(
          "stream_timeout",
          [](const std::shared_ptr<Watchdog>& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self->stream_timeout(
                currentStream(), timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def(
          "stream_completed",
          [](const std::shared_ptr<Watchdog>& self,
             std::function<void()> callback) {
            return self->stream_completed(
                currentStream(), wrapCallback(std::move(callback)));
          },
          py::arg("callback"))
      .def(
          "op_timeout",
          [](const std::shared_ptr<Watchdog>& self,
             std::chrono::milliseconds timeout,
             std::function<void()> callback) {
            return self->op_timeout(
                currentStream(), timeout, wrapCallback(std::move(callback)));
          },
          py::arg("timeout"),
          py::arg("callback"))
      .def("num_active_stream_timeouts", &Watchdog::numActiveStreamTimeouts);
}

} // namespace c10d::watchdog

#else // TORCH_USE_LIBUV

namespace c10d::watchdog {

// Without the libuv backend the submodule is not registered; importing
// torch._C._distributed_c10d_watchdog then raises ImportError.
void initWatchdogBindings(pybind11::module& /*module*/) {}

} // namespace c10d::watchdog

#endif // TORCH_USE_LIBUV
