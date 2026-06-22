#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include <c10/core/Event.h>
#include <c10/core/Stream.h>
#include <c10/macros/Export.h>

// The watchdog is only built with the libuv timer backend. Without it the types
// below are not declared, so any use is a compile error (and the Python
// bindings are not registered, so importing them raises ImportError).
#ifdef TORCH_USE_LIBUV

namespace c10d::watchdog {

// Callback invoked from the watchdog's timer loop thread. Callbacks must not
// block, since all timeouts are serviced on that single thread.
using Callback = std::function<void()>;

class Watchdog;

// Handle to a registered monitor. cancel() removes it and is idempotent.
class TORCH_API Handle {
 public:
  Handle() = default;
  void cancel() const;

 private:
  friend class Watchdog;
  Handle(std::weak_ptr<Watchdog> watchdog, uint64_t id)
      : watchdog_(std::move(watchdog)), id_(id) {}

  std::weak_ptr<Watchdog> watchdog_;
  uint64_t id_{0};
};

// A process-wide timer/timeout service backed by a libuv event loop running on
// a dedicated background thread.
//
// This is the interface; the implementation (which owns the libuv loop) derives
// from it and is created via makeWatchdog(). A single global instance is
// available via singleton(); makeWatchdog() additionally yields isolated
// instances, which tests use. Instances must be owned by a std::shared_ptr so
// that the handles they hand out can safely refer back to them.
//
// cpu_timeout, stream_timeout and stream_completed are the reusable primitives;
// op_timeout is a composition of them.
class TORCH_API Watchdog : public std::enable_shared_from_this<Watchdog> {
 public:
  virtual ~Watchdog() = default;
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  Watchdog(Watchdog&&) = delete;
  Watchdog& operator=(Watchdog&&) = delete;

  // Process-wide instance. Intentionally leaked so the background thread is
  // never joined during interpreter shutdown.
  static const std::shared_ptr<Watchdog>& singleton();

  // Fire callback after timeout elapses, unless the returned handle is
  // cancelled first.
  virtual Handle cpu_timeout(
      std::chrono::milliseconds timeout,
      Callback callback) = 0;

  // Record an event on stream now and fire callback if the work enqueued up to
  // this point has not completed within timeout. The monitor removes itself
  // once the event completes (success) or the callback fires. cancel() stops
  // it.
  virtual Handle stream_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback) = 0;

  // Record an event on stream now and fire callback once the work enqueued up
  // to this point has completed. The monitor removes itself after firing.
  // cancel() stops it.
  virtual Handle stream_completed(c10::Stream stream, Callback callback) = 0;

  // Monitor a device operation by composing the primitives above: a cpu_timeout
  // bounds the launch, and once the start event completes (detected via
  // stream_completed) its callback enqueues a stream_timeout that bounds the
  // completion. callback fires at most once. The returned handle is the launch
  // timer; cancel it once the launch (e.g. the guarded enqueue) is done -- the
  // completion monitoring continues independently.
  Handle op_timeout(
      c10::Stream stream,
      std::chrono::milliseconds timeout,
      Callback callback);

  // Number of monitors currently active. Primarily for tests.
  virtual size_t numActiveStreamTimeouts() const = 0;

 protected:
  Watchdog() = default;
  // Constructs a Handle bound to this watchdog. Available to implementations
  // since Handle's constructor is private to Watchdog.
  Handle makeHandle(uint64_t id);

 private:
  friend class Handle;
  virtual void cancel(uint64_t id) = 0;
};

// Creates a new watchdog instance. pollInterval controls how often device
// events are polled for stream/op timeouts. Primarily for tests that want an
// isolated instance (and a short interval); production code should use
// singleton().
TORCH_API std::unique_ptr<Watchdog> makeWatchdog(
    std::chrono::milliseconds pollInterval = std::chrono::seconds(1));

} // namespace c10d::watchdog

#endif // TORCH_USE_LIBUV
