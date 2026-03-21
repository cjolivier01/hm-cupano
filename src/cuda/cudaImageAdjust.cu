#include "cudaImageAdjust.cuh"

#include <cupano/gpu/gpu_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>
#include <cstdlib>

//------------------------------------------------------------------------------
// CUDA Kernel: adjustColorKernelBatch
//
// This kernel applies a per-pixel adjustment to a batch of images stored as a
// contiguous array of pixels of type T. The total number of pixels is (batchSize * width * height).
// Each thread adjusts one pixel using the appropriate PixelAdjuster.
//------------------------------------------------------------------------------
template <typename T>
__global__ void adjustColorKernelBatch(T* image, int totalPixels, const float3& adjustment) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < totalPixels) {
    image[idx] = PixelAdjuster<T>::adjust(image[idx], adjustment);
  }
}

//------------------------------------------------------------------------------
// Host Function: adjustImageCudaBatch
//
// Launches the CUDA kernel for a batch of images. The images are assumed to be
// stored consecutively in memory. Each image has dimensions (width x height) and
// there are batchSize images.
//------------------------------------------------------------------------------
template <typename T>
void adjustImageCudaBatch(T* d_image, int batchSize, int width, int height, const float3& adjustment) {
  int totalPixels = batchSize * width * height;
  int threadsPerBlock = 256;
  int blocks = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;
  adjustColorKernelBatch<T><<<blocks, threadsPerBlock>>>(d_image, totalPixels, adjustment);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

//------------------------------------------------------------------------------
// Macro for template instantiation
//
// Use this macro to instantiate adjustImageCudaBatch for a given pixel type.
// For example:
//
//    INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(uchar3)
//    INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(float4)
//
#define INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(TYPE) \
  template void adjustImageCudaBatch<TYPE>(       \
      TYPE * d_image, int batchSize, int width, int height, const float3& adjustment);

// Instantiate the template for several common types.
INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(uchar3)
INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(uchar4)
INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(float3)
INSTANTIATE_ADJUST_IMAGE_CUDA_BATCH(float4)
