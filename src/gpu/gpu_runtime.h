// Lightweight CUDA/HIP runtime compatibility layer for this project.
// Selects CUDA by default; define USE_HIP or compile under HIP to use ROCm.
#pragma once

// Backend detection
#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)
  #define GPU_BACKEND_HIP 1
  #define GPU_BACKEND_CUDA 0
#else
  #define GPU_BACKEND_HIP 0
  #define GPU_BACKEND_CUDA 1
#endif

#if GPU_BACKEND_HIP
  // HIP includes
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
  // BF16 support (best-effort)
  #if defined(__has_include)
    #if __has_include(<hip/hip_bfloat16.h>)
      #include <hip/hip_bfloat16.h>
      #define GPU_HAS_BF16 1
      // Some ROCm versions use hip_bfloat16 (public type)
      using gpu_bfloat16 = hip_bfloat16;
    #else
      #define GPU_HAS_BF16 0
    #endif
  #else
    #define GPU_HAS_BF16 0
  #endif

  // Map common CUDA symbols to HIP so most of the code can stay unchanged
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaStream_t hipStream_t
  #define cudaEvent_t hipEvent_t
  #define cudaDeviceProp hipDeviceProp_t
  #define cudaMemcpyKind hipMemcpyKind

  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define cudaMemcpyDefault hipMemcpyDefault

  #define cudaGetDeviceCount hipGetDeviceCount
  #define cudaGetDeviceProperties hipGetDeviceProperties
  #define cudaSetDevice hipSetDevice
  #define cudaGetLastError hipGetLastError
  #define cudaPeekAtLastError hipPeekAtLastError
  #define cudaGetErrorString hipGetErrorString
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaStreamCreate hipStreamCreate
  #define cudaStreamDestroy hipStreamDestroy
  #define cudaStreamSynchronize hipStreamSynchronize
  #define cudaEventCreate hipEventCreate
  #define cudaEventCreateWithFlags hipEventCreateWithFlags
  #define cudaEventDestroy hipEventDestroy
  #define cudaEventRecord hipEventRecord
  #define cudaEventSynchronize hipEventSynchronize
  #define cudaEventElapsedTime hipEventElapsedTime

  #define cudaMalloc hipMalloc
  #define cudaFree hipFree
  #define cudaMemset hipMemset
  #define cudaMemcpy hipMemcpy
  #define cudaMemcpyAsync hipMemcpyAsync

  // Optional: macro to launch kernels with hipLaunchKernelGGL if needed
  #define GPU_LAUNCH_KERNEL(func, grid, block, shared, stream, ...) \
      hipLaunchKernelGGL(func, grid, block, shared, stream, ##__VA_ARGS__)

#else  // GPU_BACKEND_CUDA

  // CUDA includes
  #include <cuda_runtime.h>
  #include <cuda_runtime_api.h>
  #include <cuda_fp16.h>

  // BF16 support for CUDA 11+
  #if defined(CUDART_VERSION) && (CUDART_VERSION >= 11000)
    #include <cuda_bf16.h>
    #define GPU_HAS_BF16 1
    using gpu_bfloat16 = __nv_bfloat16;
  #else
    #define GPU_HAS_BF16 0
  #endif

  #define GPU_LAUNCH_KERNEL(func, grid, block, shared, stream, ...) \
      func<<<(grid), (block), (shared), (stream)>>>(__VA_ARGS__)

#endif  // backend select
