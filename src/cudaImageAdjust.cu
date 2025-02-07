#include "cudaImageAdjust.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

//------------------------------------------------------------------------------
// Device helper: clampf
//
// Clamps a float value between minVal and maxVal.
//------------------------------------------------------------------------------
__device__ float clampf(float val, float minVal, float maxVal) {
  return fminf(maxVal, fmaxf(minVal, val));
}

// Specialization for uchar3 (3-channel 8-bit pixel)
template <>
struct PixelAdjuster<uchar3> {
  __device__ static uchar3 adjust(const uchar3& pixel, const float3& adjustment) {
    uchar3 out;
    out.x = static_cast<unsigned char>(clampf(float(pixel.x) + adjustment.x, 0.0f, 255.0f));
    out.y = static_cast<unsigned char>(clampf(float(pixel.y) + adjustment.y, 0.0f, 255.0f));
    out.z = static_cast<unsigned char>(clampf(float(pixel.z) + adjustment.z, 0.0f, 255.0f));
    return out;
  }
};

// Specialization for uchar4 (4-channel 8-bit pixel)
template <>
struct PixelAdjuster<uchar4> {
  __device__ static uchar4 adjust(const uchar4& pixel, const float3& adjustment) {
    uchar4 out;
    out.x = static_cast<unsigned char>(clampf(float(pixel.x) + adjustment.x, 0.0f, 255.0f));
    out.y = static_cast<unsigned char>(clampf(float(pixel.y) + adjustment.y, 0.0f, 255.0f));
    out.z = static_cast<unsigned char>(clampf(float(pixel.z) + adjustment.z, 0.0f, 255.0f));
    out.w = pixel.w; // Preserve alpha.
    return out;
  }
};

// Specialization for float3 (3-channel floating-point pixel)
template <>
struct PixelAdjuster<float3> {
  __device__ static float3 adjust(const float3& pixel, const float3& adjustment) {
    float3 out;
    out.x = pixel.x + adjustment.x;
    out.y = pixel.y + adjustment.y;
    out.z = pixel.z + adjustment.z;
    return out;
  }
};

// Specialization for float4 (4-channel floating-point pixel)
template <>
struct PixelAdjuster<float4> {
  __device__ static float4 adjust(const float4& pixel, const float3& adjustment) {
    float4 out;
    out.x = pixel.x + adjustment.x;
    out.y = pixel.y + adjustment.y;
    out.z = pixel.z + adjustment.z;
    out.w = pixel.w; // Preserve alpha.
    return out;
  }
};

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
