#pragma once

#include "cupano/gpu/gpu_runtime.h"

#if GPU_BACKEND_HIP
  // HIP-OpenGL interop header
  #include <hip/hip_gl_interop.h>
  // Map CUDA GL interop names if needed
  #define cudaGraphicsResource hipGraphicsResource
  #define cudaGraphicsGLRegisterImage hipGraphicsGLRegisterImage
  #define cudaGraphicsGLRegisterBuffer hipGraphicsGLRegisterBuffer
  #define cudaGraphicsMapResources hipGraphicsMapResources
  #define cudaGraphicsUnmapResources hipGraphicsUnmapResources
  #define cudaGraphicsSubResourceGetMappedArray hipGraphicsSubResourceGetMappedArray
  #define cudaGraphicsResourceGetMappedPointer hipGraphicsResourceGetMappedPointer
  #define cudaGraphicsUnregisterResource hipGraphicsUnregisterResource
  #define cudaGraphicsRegisterFlagsNone hipGraphicsRegisterFlagsNone
  #define cudaGraphicsMapFlagsNone hipGraphicsMapFlagsNone
  #define cudaGraphicsRegisterFlagsWriteDiscard hipGraphicsRegisterFlagsWriteDiscard
  #define cudaArray_t hipArray_t
  #define cudaMemcpy2DToArray hipMemcpy2DToArray
#elif GPU_BACKEND_VULKAN
static constexpr unsigned int cudaGraphicsRegisterFlagsNone = 0;
static constexpr unsigned int cudaGraphicsMapFlagsNone = 0;
static constexpr unsigned int cudaGraphicsRegisterFlagsWriteDiscard = 1;

inline cudaError_t cudaGraphicsGLRegisterImage(cudaGraphicsResource**, unsigned int, unsigned int, unsigned int) {
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsGLRegisterBuffer(cudaGraphicsResource**, unsigned int, unsigned int) {
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsMapResources(int, cudaGraphicsResource**, cudaStream_t = nullptr) {
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsUnmapResources(int, cudaGraphicsResource**, cudaStream_t = nullptr) {
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsSubResourceGetMappedArray(cudaArray_t* arr, cudaGraphicsResource*, unsigned int, unsigned int) {
  if (arr) {
    *arr = nullptr;
  }
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsResourceGetMappedPointer(void**, size_t*, cudaGraphicsResource*) {
  return cudaErrorNotSupported;
}

inline cudaError_t cudaGraphicsUnregisterResource(cudaGraphicsResource*) {
  return cudaErrorNotSupported;
}
#else
  // CUDA-OpenGL interop header
  #include <cuda_gl_interop.h>
#endif
