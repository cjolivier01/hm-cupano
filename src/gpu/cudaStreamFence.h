#pragma once

#include <cupano/gpu/gpu_runtime.h>

namespace hm {
namespace gpu {

// A context may outlive its caller's stream. Retain a completion event instead
// of a stream handle, then order destructor frees on the default stream.
// Calls using the context must still be serialized by the caller.
class CudaStreamFence {
 public:
  CudaStreamFence() = default;
  CudaStreamFence(const CudaStreamFence&) = delete;
  CudaStreamFence& operator=(const CudaStreamFence&) = delete;

  ~CudaStreamFence() {
    if (event_)
      cudaEventDestroy(event_);
  }

  cudaError_t initialize() {
    return event_ ? cudaSuccess : cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
  }

  void order_cleanup() const {
    if (event_ && cudaStreamWaitEvent(0, event_, 0) != cudaSuccess)
      cudaDeviceSynchronize();
  }

  class RecordOnExit {
   public:
    RecordOnExit(CudaStreamFence& fence, cudaStream_t stream) : fence_(fence), stream_(stream) {}
    ~RecordOnExit() {
      // Also cover allocations and kernels queued before an error return.
      if (cudaEventRecord(fence_.event_, stream_) != cudaSuccess)
        cudaDeviceSynchronize();
    }

   private:
    CudaStreamFence& fence_;
    cudaStream_t stream_;
  };

 private:
  cudaEvent_t event_{nullptr};
};

} // namespace gpu
} // namespace hm
