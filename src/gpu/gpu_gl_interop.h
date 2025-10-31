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
#else
  // CUDA-OpenGL interop header
  #include <cuda_gl_interop.h>
#endif

