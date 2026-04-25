#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
#ifndef __shared__
#define __shared__
#endif
#ifndef __forceinline__
#define __forceinline__ inline
#endif

struct __half {
  std::uint16_t x;
};

struct uchar1 {
  unsigned char x;
} __attribute__((packed));
struct uchar3 {
  unsigned char x, y, z;
} __attribute__((packed));
struct uchar4 {
  unsigned char x, y, z, w;
} __attribute__((packed));

struct ushort3 {
  unsigned short x, y, z;
} __attribute__((packed));
struct ushort4 {
  unsigned short x, y, z, w;
} __attribute__((packed));

struct int2 {
  int x, y;
};
struct int3 {
  int x, y, z;
};
struct int4 {
  int x, y, z, w;
};

struct float1 {
  float x;
};
struct float3 {
  float x, y, z;
};
struct float4 {
  float x, y, z, w;
};

inline float3 make_float3(float x, float y, float z) {
  return float3{x, y, z};
}
inline float4 make_float4(float x, float y, float z, float w) {
  return float4{x, y, z, w};
}
inline int2 make_int2(int x, int y) {
  return int2{x, y};
}

struct dim3 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  constexpr dim3(unsigned int x_ = 1, unsigned int y_ = 1, unsigned int z_ = 1) : x(x_), y(y_), z(z_) {}
};

enum cudaError_t {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorUnknown = 30,
  cudaErrorNotSupported = 801,
  cudaErrorAssert = 710,
  cudaErrorFileNotFound = 301,
  cudaErrorInvalidDevicePointer = 17,
};

enum cudaMemcpyKind {
  cudaMemcpyHostToHost = 0,
  cudaMemcpyHostToDevice = 1,
  cudaMemcpyDeviceToHost = 2,
  cudaMemcpyDeviceToDevice = 3,
  cudaMemcpyDefault = 4,
};

using cudaStream_t = void*;

struct cudaFakeEvent {
  std::chrono::steady_clock::time_point time;
};
using cudaEvent_t = cudaFakeEvent*;

using cudaArray_t = void*;
using cudaGraphicsResource = void*;

struct cudaDeviceProp {
  char name[256];
  std::size_t totalGlobalMem;
  int major;
  int minor;
};

inline cudaError_t& cuda_last_error_ref() {
  static thread_local cudaError_t err = cudaSuccess;
  return err;
}

inline void cuda_set_last_error(cudaError_t err) {
  cuda_last_error_ref() = err;
}

inline const char* cudaGetErrorString(cudaError_t err) {
  switch (err) {
    case cudaSuccess:
      return "cudaSuccess";
    case cudaErrorInvalidValue:
      return "cudaErrorInvalidValue";
    case cudaErrorMemoryAllocation:
      return "cudaErrorMemoryAllocation";
    case cudaErrorUnknown:
      return "cudaErrorUnknown";
    case cudaErrorNotSupported:
      return "cudaErrorNotSupported";
    case cudaErrorAssert:
      return "cudaErrorAssert";
    case cudaErrorFileNotFound:
      return "cudaErrorFileNotFound";
    case cudaErrorInvalidDevicePointer:
      return "cudaErrorInvalidDevicePointer";
    default:
      return "cudaError";
  }
}

inline cudaError_t cudaGetLastError() {
  cudaError_t err = cuda_last_error_ref();
  cuda_last_error_ref() = cudaSuccess;
  return err;
}

inline cudaError_t cudaPeekAtLastError() {
  return cuda_last_error_ref();
}

inline cudaError_t cudaGetDeviceCount(int* count) {
  if (!count) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  *count = 1;
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* prop, int) {
  if (!prop) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  std::snprintf(prop->name, sizeof(prop->name), "%s", "Vulkan-CPU-Shim");
  prop->totalGlobalMem = 0;
  prop->major = 1;
  prop->minor = 0;
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaSetDevice(int) {
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaDeviceSynchronize() {
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaStreamCreate(cudaStream_t* stream) {
  if (!stream) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  *stream = reinterpret_cast<cudaStream_t>(1);
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaStreamDestroy(cudaStream_t) {
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaStreamSynchronize(cudaStream_t) {
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaEventCreate(cudaEvent_t* event) {
  if (!event) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  *event = new cudaFakeEvent{std::chrono::steady_clock::now()};
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int) {
  return cudaEventCreate(event);
}

inline cudaError_t cudaEventDestroy(cudaEvent_t event) {
  delete event;
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t) {
  if (!event) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  event->time = std::chrono::steady_clock::now();
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaEventSynchronize(cudaEvent_t) {
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end) {
  if (!ms || !start || !end) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  *ms = std::chrono::duration<float, std::milli>(end->time - start->time).count();
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaMalloc(void** ptr, std::size_t size) {
  if (!ptr || size == 0) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  *ptr = std::malloc(size);
  if (!*ptr) {
    cuda_set_last_error(cudaErrorMemoryAllocation);
    return cudaErrorMemoryAllocation;
  }
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

template <typename T>
inline cudaError_t cudaMalloc(T** ptr, std::size_t size) {
  return cudaMalloc(reinterpret_cast<void**>(ptr), size);
}

inline cudaError_t cudaFree(void* ptr) {
  std::free(ptr);
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

template <typename T>
inline cudaError_t cudaFree(T* ptr) {
  return cudaFree(reinterpret_cast<void*>(ptr));
}

inline cudaError_t cudaMemset(void* dst, int value, std::size_t count) {
  if (!dst) {
    cuda_set_last_error(cudaErrorInvalidDevicePointer);
    return cudaErrorInvalidDevicePointer;
  }
  std::memset(dst, value, count);
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaMemsetAsync(void* dst, int value, std::size_t count, cudaStream_t) {
  return cudaMemset(dst, value, count);
}

inline cudaError_t cudaMemcpy(void* dst, const void* src, std::size_t count, cudaMemcpyKind) {
  if (!dst || !src) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  std::memcpy(dst, src, count);
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaMemcpyAsync(void* dst, const void* src, std::size_t count, cudaMemcpyKind kind, cudaStream_t) {
  return cudaMemcpy(dst, src, count, kind);
}

inline cudaError_t cudaMemcpy2D(
    void* dst,
    std::size_t dpitch,
    const void* src,
    std::size_t spitch,
    std::size_t width,
    std::size_t height,
    cudaMemcpyKind) {
  if (!dst || !src) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  auto* d = static_cast<std::uint8_t*>(dst);
  auto* s = static_cast<const std::uint8_t*>(src);
  for (std::size_t y = 0; y < height; ++y) {
    std::memcpy(d + y * dpitch, s + y * spitch, width);
  }
  cuda_set_last_error(cudaSuccess);
  return cudaSuccess;
}

inline cudaError_t cudaMemcpy2DAsync(
    void* dst,
    std::size_t dpitch,
    const void* src,
    std::size_t spitch,
    std::size_t width,
    std::size_t height,
    cudaMemcpyKind kind,
    cudaStream_t) {
  return cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind);
}

inline cudaError_t cudaMemcpy2DFromArray(
    void* dst,
    std::size_t dpitch,
    const cudaArray_t src,
    std::size_t wOffset,
    std::size_t hOffset,
    std::size_t width,
    std::size_t height,
    cudaMemcpyKind kind) {
  if (!dst || !src) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  // This shim models cudaArray_t as linear memory without explicit stride metadata.
  // Treat copied rows as tightly packed by width bytes.
  const std::size_t array_pitch = width;
  const auto* src_ptr = static_cast<const std::uint8_t*>(src) + hOffset * array_pitch + wOffset;
  return cudaMemcpy2D(dst, dpitch, src_ptr, array_pitch, width, height, kind);
}

inline cudaError_t cudaMemcpy2DFromArrayAsync(
    void* dst,
    std::size_t dpitch,
    const cudaArray_t src,
    std::size_t wOffset,
    std::size_t hOffset,
    std::size_t width,
    std::size_t height,
    cudaMemcpyKind kind,
    cudaStream_t) {
  return cudaMemcpy2DFromArray(dst, dpitch, src, wOffset, hOffset, width, height, kind);
}

inline cudaError_t cudaMemcpy2DToArray(
    cudaArray_t dst,
    std::size_t wOffset,
    std::size_t hOffset,
    const void* src,
    std::size_t spitch,
    std::size_t width,
    std::size_t height,
    cudaMemcpyKind kind) {
  if (!dst || !src) {
    cuda_set_last_error(cudaErrorInvalidValue);
    return cudaErrorInvalidValue;
  }
  // This shim models cudaArray_t as linear memory without explicit stride metadata.
  // Treat copied rows as tightly packed by width bytes.
  const std::size_t array_pitch = width;
  auto* dst_ptr = static_cast<std::uint8_t*>(dst) + hOffset * array_pitch + wOffset;
  return cudaMemcpy2D(dst_ptr, array_pitch, src, spitch, width, height, kind);
}
