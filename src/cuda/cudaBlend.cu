#include "cudaBlend.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

// =============================================================================
// Macro to check CUDA calls and return on error.
#define CUDA_CHECK(call)                                                                          \
  do {                                                                                            \
    cudaError_t _err = (call);                                                                    \
    if (_err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
      return _err;                                                                                \
    }                                                                                             \
  } while (0)

// =============================================================================
// Templated CUDA Kernels for Batched Laplacian Blending
// =============================================================================

// -----------------------------------------------------------------------------
// Fused downsample kernel for two RGB images.
// Instead of launching separate kernels for image1 and image2, we fuse them
// into one kernel call. Each output pixel is computed by averaging a 2x2 block.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void FusedBatchedDownsampleKernelRGB(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int inWidth,
    int inHeight,
    T* __restrict__ output1,
    T* __restrict__ output2,
    int outWidth,
    int outHeight,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * 3;
  int outImageSize = outWidth * outHeight * 3;
  const T* inImage1 = input1 + b * inImageSize;
  const T* inImage2 = input2 + b * inImageSize;
  T* outImage1 = output1 + b * outImageSize;
  T* outImage2 = output2 + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;
  float sumR1 = 0.0f, sumG1 = 0.0f, sumB1 = 0.0f;
  float sumR2 = 0.0f, sumG2 = 0.0f, sumB2 = 0.0f;
  int count = 0;
#pragma unroll
  for (int dy = 0; dy < 2; dy++) {
#pragma unroll
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * 3;
        sumR1 += static_cast<float>(inImage1[idx + 0]);
        sumG1 += static_cast<float>(inImage1[idx + 1]);
        sumB1 += static_cast<float>(inImage1[idx + 2]);
        sumR2 += static_cast<float>(inImage2[idx + 0]);
        sumG2 += static_cast<float>(inImage2[idx + 1]);
        sumB2 += static_cast<float>(inImage2[idx + 2]);
        count++;
      }
    }
  }
  int outIdx = (y * outWidth + x) * 3;
  outImage1[outIdx + 0] = static_cast<T>(sumR1 / count);
  outImage1[outIdx + 1] = static_cast<T>(sumG1 / count);
  outImage1[outIdx + 2] = static_cast<T>(sumB1 / count);
  outImage2[outIdx + 0] = static_cast<T>(sumR2 / count);
  outImage2[outIdx + 1] = static_cast<T>(sumG2 / count);
  outImage2[outIdx + 2] = static_cast<T>(sumB2 / count);
}

// -----------------------------------------------------------------------------
// Batched downsample kernel for a single image with __restrict__ and unrolling.
// This kernel is kept in case separate launches are desired.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void BatchedDownsampleKernelRGB(
    const T* __restrict__ input,
    int inWidth,
    int inHeight,
    T* __restrict__ output,
    int outWidth,
    int outHeight,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * 3;
  int outImageSize = outWidth * outHeight * 3;
  const T* inImage = input + b * inImageSize;
  T* outImage = output + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;
  float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
  int count = 0;
#pragma unroll
  for (int dy = 0; dy < 2; dy++) {
#pragma unroll
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * 3;
        sumR += static_cast<float>(inImage[idx + 0]);
        sumG += static_cast<float>(inImage[idx + 1]);
        sumB += static_cast<float>(inImage[idx + 2]);
        count++;
      }
    }
  }
  int outIdx = (y * outWidth + x) * 3;
  outImage[outIdx + 0] = static_cast<T>(sumR / count);
  outImage[outIdx + 1] = static_cast<T>(sumG / count);
  outImage[outIdx + 2] = static_cast<T>(sumB / count);
}

// -----------------------------------------------------------------------------
// Batched downsample kernel for a single-channel mask with __restrict__ and unrolling.
// -----------------------------------------------------------------------------
template <typename T>
__global__ void BatchedDownsampleKernelMask(
    const T* __restrict__ input,
    int inWidth,
    int inHeight,
    T* __restrict__ output,
    int outWidth,
    int outHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inX = x * 2;
  int inY = y * 2;
  float sum = 0.0f;
  int count = 0;
#pragma unroll
  for (int dy = 0; dy < 2; dy++) {
#pragma unroll
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        sum += static_cast<float>(input[iy * inWidth + ix]);
        count++;
      }
    }
  }
  output[y * outWidth + x] = static_cast<T>(sum / count);
}

// -----------------------------------------------------------------------------
// Batched upsample kernel for RGB images using bilinear interpolation.
// -----------------------------------------------------------------------------
template <typename T, typename F_T>
__global__ void BatchedUpsampleKernelRGB(
    const T* __restrict__ input,
    int inWidth,
    int inHeight,
    T* __restrict__ output,
    int outWidth,
    int outHeight,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * 3;
  int outImageSize = outWidth * outHeight * 3;
  const T* inImage = input + b * inImageSize;
  T* outImage = output + b * outImageSize;

  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  F_T dx = gx - gxi;
  F_T dy = gy - gyi;
  int gxi1 = min(gxi + 1, inWidth - 1);
  int gyi1 = min(gyi + 1, inHeight - 1);

  int idx00 = (gyi * inWidth + gxi) * 3;
  int idx10 = (gyi * inWidth + gxi1) * 3;
  int idx01 = (gyi1 * inWidth + gxi) * 3;
  int idx11 = (gyi1 * inWidth + gxi1) * 3;

  F_T outR, outG, outB;
  { // R channel interpolation.
    F_T val00 = static_cast<F_T>(inImage[idx00 + 0]);
    F_T val10 = static_cast<F_T>(inImage[idx10 + 0]);
    F_T val01 = static_cast<F_T>(inImage[idx01 + 0]);
    F_T val11 = static_cast<F_T>(inImage[idx11 + 0]);
    F_T val0 = val00 * (1.0f - dx) + val10 * dx;
    F_T val1 = val01 * (1.0f - dx) + val11 * dx;
    outR = val0 * (1.0f - dy) + val1 * dy;
  }
  { // G channel interpolation.
    F_T val00 = static_cast<F_T>(inImage[idx00 + 1]);
    F_T val10 = static_cast<F_T>(inImage[idx10 + 1]);
    F_T val01 = static_cast<F_T>(inImage[idx01 + 1]);
    F_T val11 = static_cast<F_T>(inImage[idx11 + 1]);
    F_T val0 = val00 * (1.0f - dx) + val10 * dx;
    F_T val1 = val01 * (1.0f - dx) + val11 * dx;
    outG = val0 * (1.0f - dy) + val1 * dy;
  }
  { // B channel interpolation.
    F_T val00 = static_cast<F_T>(inImage[idx00 + 2]);
    F_T val10 = static_cast<F_T>(inImage[idx10 + 2]);
    F_T val01 = static_cast<F_T>(inImage[idx01 + 2]);
    F_T val11 = static_cast<F_T>(inImage[idx11 + 2]);
    F_T val0 = val00 * (1.0f - dx) + val10 * dx;
    F_T val1 = val01 * (1.0f - dx) + val11 * dx;
    outB = val0 * (1.0f - dy) + val1 * dy;
  }
  int idxOut = (y * outWidth + x) * 3;
  outImage[idxOut + 0] = static_cast<T>(outR);
  outImage[idxOut + 1] = static_cast<T>(outG);
  outImage[idxOut + 2] = static_cast<T>(outB);
}

// -----------------------------------------------------------------------------
// Batched computation of the Laplacian for RGB images.
// For each pixel, compute Laplacian = Gaussian_high - upsample(Gaussian_low).
// -----------------------------------------------------------------------------
template <typename T, typename F_T>
__global__ void BatchedComputeLaplacianKernelRGB(
    const T* __restrict__ gaussHigh,
    int highWidth,
    int highHeight,
    const T* __restrict__ gaussLow,
    int lowWidth,
    int lowHeight,
    T* __restrict__ laplacian,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int highImageSize = highWidth * highHeight * 3;
  int lowImageSize = lowWidth * lowHeight * 3;
  const T* highImage = gaussHigh + b * highImageSize;
  const T* lowImage = gaussLow + b * lowImageSize;
  T* lapImage = laplacian + b * highImageSize;

  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idx00 = (gyi * lowWidth + gxi) * 3;
  int idx10 = (gyi * lowWidth + gxi1) * 3;
  int idx01 = (gyi1 * lowWidth + gxi) * 3;
  int idx11 = (gyi1 * lowWidth + gxi1) * 3;

  const F_T F_ONE = static_cast<F_T>(1.0);
  F_T upR, upG, upB;
  { // R channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 0]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 0]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 0]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 0]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upR = val0 * (F_ONE - dy) + val1 * dy;
  }
  { // G channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 1]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 1]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 1]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 1]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upG = val0 * (F_ONE - dy) + val1 * dy;
  }
  { // B channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 2]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 2]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 2]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 2]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upB = val0 * (F_ONE - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  lapImage[idxHigh + 0] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + 0]) - upR);
  lapImage[idxHigh + 1] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + 1]) - upG);
  lapImage[idxHigh + 2] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + 2]) - upB);
}

// -----------------------------------------------------------------------------
// Batched blend kernel for RGB images.
// Blends two Laplacian pyramid levels using a shared single-channel mask.
// -----------------------------------------------------------------------------
template <typename T, typename F_T>
__global__ void BatchedBlendKernelRGB(
    const T* __restrict__ lap1,
    const T* __restrict__ lap2,
    const T* __restrict__ mask,
    T* __restrict__ blended,
    int width,
    int height,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int imageSizeRGB = width * height * 3;
  const T* lap1Image = lap1 + b * imageSizeRGB;
  const T* lap2Image = lap2 + b * imageSizeRGB;
  // The mask is shared (non-batched)
  const T* maskImage = mask;
  T* blendImage = blended + b * imageSizeRGB;

  const F_T F_ONE = static_cast<F_T>(1.0);
  int idx = (y * width + x) * 3;
  F_T m = static_cast<F_T>(maskImage[y * width + x]);
  F_T mm1 = F_ONE - m;
  blendImage[idx + 0] =
      static_cast<T>(m * static_cast<F_T>(lap1Image[idx + 0]) + mm1 * static_cast<F_T>(lap2Image[idx + 0]));
  blendImage[idx + 1] =
      static_cast<T>(m * static_cast<F_T>(lap1Image[idx + 1]) + mm1 * static_cast<F_T>(lap2Image[idx + 1]));
  blendImage[idx + 2] =
      static_cast<T>(m * static_cast<F_T>(lap1Image[idx + 2]) + mm1 * static_cast<F_T>(lap2Image[idx + 2]));
}

// -----------------------------------------------------------------------------
// Batched reconstruction kernel for RGB images.
// Reconstructs the high-resolution image by upsampling the lower-resolution image and
// adding the blended Laplacian details.
// -----------------------------------------------------------------------------
template <typename T, typename F_T>
__global__ void BatchedReconstructKernelRGB(
    const T* __restrict__ lowerRes,
    int lowWidth,
    int lowHeight,
    const T* __restrict__ lap,
    int highWidth,
    int highHeight,
    T* __restrict__ reconstruction,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int lowImageSize = lowWidth * lowHeight * 3;
  int highImageSize = highWidth * highHeight * 3;
  const T* lowImage = lowerRes + b * lowImageSize;
  const T* lapImage = lap + b * highImageSize;
  T* reconImage = reconstruction + b * highImageSize;

  const F_T F_ONE = static_cast<F_T>(1.0);
  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idx00 = (gyi * lowWidth + gxi) * 3;
  int idx10 = (gyi * lowWidth + gxi1) * 3;
  int idx01 = (gyi1 * lowWidth + gxi) * 3;
  int idx11 = (gyi1 * lowWidth + gxi1) * 3;

  F_T upR, upG, upB;
  { // Reconstruct R channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 0]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 0]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 0]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 0]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upR = val0 * (F_ONE - dy) + val1 * dy;
  }
  { // Reconstruct G channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 1]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 1]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 1]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 1]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upG = val0 * (F_ONE - dy) + val1 * dy;
  }
  { // Reconstruct B channel.
    F_T val00 = static_cast<F_T>(lowImage[idx00 + 2]);
    F_T val10 = static_cast<F_T>(lowImage[idx10 + 2]);
    F_T val01 = static_cast<F_T>(lowImage[idx01 + 2]);
    F_T val11 = static_cast<F_T>(lowImage[idx11 + 2]);
    F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
    F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
    upB = val0 * (F_ONE - dy) + val1 * dy;
  }
  int idxHigh = (y * highWidth + x) * 3;
  reconImage[idxHigh + 0] = static_cast<T>(upR + static_cast<F_T>(lapImage[idxHigh + 0]));
  reconImage[idxHigh + 1] = static_cast<T>(upG + static_cast<F_T>(lapImage[idxHigh + 1]));
  reconImage[idxHigh + 2] = static_cast<T>(upB + static_cast<F_T>(lapImage[idxHigh + 2]));
}

// =============================================================================
// Templated Host Functions: Batched Laplacian Blending
// =============================================================================

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlend(
    const T* h_image1,
    const T* h_image2,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int numLevels,
    int batchSize,
    cudaStream_t stream) {
  // For RGB images (3 channels) and a single-channel mask.
  size_t imageSize = imageWidth * imageHeight * 3 * sizeof(T);
  size_t maskSize = imageWidth * imageHeight * sizeof(T);

  // Allocate device memory for level-0 Gaussian pyramid images.
  std::vector<T*> d_gauss1(numLevels);
  std::vector<T*> d_gauss2(numLevels);
  std::vector<T*> d_maskPyr(numLevels);
  std::vector<T*> d_lap1(numLevels);
  std::vector<T*> d_lap2(numLevels);
  std::vector<T*> d_blend(numLevels);

  std::vector<int> widths(numLevels), heights(numLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int i = 1; i < numLevels; i++) {
    widths[i] = (widths[i - 1] + 1) / 2;
    heights[i] = (heights[i - 1] + 1) / 2;
  }

  // Allocate level 0 arrays and copy input data from host to device.
  size_t sizeRGB0 = widths[0] * heights[0] * 3 * batchSize * sizeof(T);
  size_t sizeMask0 = widths[0] * heights[0] * sizeof(T);
  cudaMalloc((void**)&d_gauss1[0], sizeRGB0);
  cudaMalloc((void**)&d_gauss2[0], sizeRGB0);
  cudaMalloc((void**)&d_maskPyr[0], sizeMask0);
  cudaMemcpyAsync(d_gauss1[0], h_image1, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_gauss2[0], h_image2, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_maskPyr[0], h_mask, maskSize, cudaMemcpyHostToDevice, stream);

  // Allocate device memory for higher pyramid levels.
  for (int level = 1; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(T);
    size_t sizeMask = widths[level] * heights[level] * sizeof(T);
    cudaMalloc((void**)&d_gauss1[level], sizeRGB);
    cudaMalloc((void**)&d_gauss2[level], sizeRGB);
    cudaMalloc((void**)&d_maskPyr[level], sizeMask);
  }

  dim3 block(16, 16, 1);

  // 1. Build Gaussian pyramids for both image sets and the shared mask.
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 gridRGB((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, batchSize);
    // Use fused downsampling for the two image sets.
    FusedBatchedDownsampleKernelRGB<T><<<gridRGB, block, 0, stream>>>(
        d_gauss1[level],
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        batchSize);
    // Downsample the shared mask (non-batched).
    {
      dim3 gridMask((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, 1);
      BatchedDownsampleKernelMask<T><<<gridMask, block, 0, stream>>>(
          d_maskPyr[level], widths[level], heights[level], d_maskPyr[level + 1], widths[level + 1], heights[level + 1]);
    }
  }

  // 2. Build Laplacian pyramids.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(T);
    cudaMalloc((void**)&d_lap1[level], sizeRGB);
    cudaMalloc((void**)&d_lap2[level], sizeRGB);
  }
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedComputeLaplacianKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        d_gauss1[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap1[level],
        batchSize);
    BatchedComputeLaplacianKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap2[level],
        batchSize);
  }
  int last = numLevels - 1;
  {
    size_t lastSize = widths[last] * heights[last] * 3 * batchSize * sizeof(T);
    cudaMemcpyAsync(d_lap1[last], d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_lap2[last], d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream);
  }

  // 3. Blend the Laplacian pyramids using the shared mask.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * 3 * batchSize * sizeof(T);
    cudaMalloc((void**)&d_blend[level], sizeRGB);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedBlendKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        d_lap1[level], d_lap2[level], d_maskPyr[level], d_blend[level], widths[level], heights[level], batchSize);
  }

  // 4. Reconstruct the final blended image.
  T* d_reconstruct = nullptr;
  cudaMalloc((void**)&d_reconstruct, widths[last] * heights[last] * 3 * batchSize * sizeof(T));
  cudaMemcpyAsync(
      d_reconstruct,
      d_blend[last],
      widths[last] * heights[last] * 3 * batchSize * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream);
  for (int level = numLevels - 2; level >= 0; level--) {
    T* d_temp = nullptr;
    size_t highSize = widths[level] * heights[level] * 3 * batchSize * sizeof(T);
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedReconstructKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        d_reconstruct,
        widths[level + 1],
        heights[level + 1],
        d_blend[level],
        widths[level],
        heights[level],
        d_temp,
        batchSize);
    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }
  cudaMemcpyAsync(h_output, d_reconstruct, imageSize * batchSize, cudaMemcpyDeviceToHost, stream);
  cudaFree(d_reconstruct);

  // Cleanup allocated device memory.
  for (int level = 0; level < numLevels; level++) {
    cudaFree(d_gauss1[level]);
    cudaFree(d_gauss2[level]);
    cudaFree(d_maskPyr[level]);
    cudaFree(d_lap1[level]);
    cudaFree(d_lap2[level]);
    cudaFree(d_blend[level]);
  }

  return cudaGetLastError();
}

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendWithContext(
    const T* d_image1,
    const T* d_image2,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext<T>& context,
    cudaStream_t stream) {
  size_t imageSize = context.imageWidth * context.imageHeight * 3 * sizeof(T);

  // If context not yet initialized, set up pyramid dimensions and allocate device memory.
  if (!context.initialized) {
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < context.numLevels; i++) {
      context.widths[i] = (context.widths[i - 1] + 1) / 2;
      context.heights[i] = (context.heights[i - 1] + 1) / 2;
      assert(context.widths[i] && context.heights[i]);
    }
    for (int level = 0; level < context.numLevels; level++) {
      size_t sizeRGB = context.widths[level] * context.heights[level] * 3 * context.batchSize * sizeof(T);
      size_t sizeMask = context.widths[level] * context.heights[level] * sizeof(T);
      assert(sizeRGB && sizeMask);
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap1[level], sizeRGB));
      context.allocation_size += sizeRGB;
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap2[level], sizeRGB));
      context.allocation_size += sizeRGB;
      CUDA_CHECK(cudaMalloc((void**)&context.d_blend[level], sizeRGB));
      context.allocation_size += sizeRGB;
      if (level > 0) {
        CUDA_CHECK(cudaMalloc((void**)&context.d_maskPyr[level], sizeMask));
        context.allocation_size += sizeMask;
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss1[level], sizeRGB));
        context.allocation_size += sizeRGB;
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss2[level], sizeRGB));
        context.allocation_size += sizeRGB;
      } else {
        context.d_maskPyr[0] = const_cast<T*>(d_mask);
        context.d_gauss1[0] = const_cast<T*>(d_image1);
        context.d_gauss2[0] = const_cast<T*>(d_image2);
      }
    }
  }

  dim3 block(16, 16, 1);

  // 1. Build Gaussian pyramids.
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 grid(
        (context.widths[level + 1] + block.x - 1) / block.x,
        (context.heights[level + 1] + block.y - 1) / block.y,
        context.batchSize);
    FusedBatchedDownsampleKernelRGB<T><<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.batchSize);
    CUDA_CHECK(cudaGetLastError());
    {
      dim3 gridMask(
          (context.widths[level + 1] + block.x - 1) / block.x, (context.heights[level + 1] + block.y - 1) / block.y, 1);
      BatchedDownsampleKernelMask<T><<<gridMask, block, 0, stream>>>(
          context.d_maskPyr[level],
          context.widths[level],
          context.heights[level],
          context.d_maskPyr[level + 1],
          context.widths[level + 1],
          context.heights[level + 1]);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // 2. Build Laplacian pyramids.
  for (int level = 0; level < context.numLevels - 1; level++) {
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedComputeLaplacianKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap1[level],
        context.batchSize);
    CUDA_CHECK(cudaGetLastError());
    BatchedComputeLaplacianKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap2[level],
        context.batchSize);
    CUDA_CHECK(cudaGetLastError());
  }
  int last = context.numLevels - 1;
  CUDA_CHECK(cudaMemcpyAsync(
      context.d_lap1[last],
      context.d_gauss1[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      context.d_lap2[last],
      context.d_gauss2[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));

  // 3. Blend the Laplacian pyramids.
  for (int level = 0; level < context.numLevels; level++) {
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedBlendKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        context.d_lap1[level],
        context.d_lap2[level],
        context.d_maskPyr[level],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        context.batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  // 4. Reconstruct the final image.
  T* d_reconstruct = nullptr;
  if (!context.initialized) {
    if (context.numLevels > 1) {
      size_t sz = context.widths[last] * context.heights[last] * 3 * sizeof(T) * context.batchSize;
      CUDA_CHECK(cudaMalloc((void**)&d_reconstruct, sz));
      context.allocation_size += sz;
      assert(last);
      assert(!context.d_resonstruct[last]);
      context.d_resonstruct[last] = d_reconstruct;
    } else {
      d_reconstruct = d_output;
      assert(last == 0);
      context.d_resonstruct[last] = d_reconstruct;
    }
  } else {
    assert(last >= 0);
    d_reconstruct = context.d_resonstruct[last];
    assert(d_reconstruct);
  }
  CUDA_CHECK(cudaMemcpyAsync(
      d_reconstruct,
      context.d_blend[last],
      context.widths[last] * context.heights[last] * 3 * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));
  for (int level = context.numLevels - 2; level >= 0; level--) {
    T* d_temp = nullptr;
    if (!context.initialized) {
      size_t highSize = context.widths[level] * context.heights[level] * 3 * sizeof(T) * context.batchSize;
      if (level > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_temp, highSize));
        context.allocation_size += highSize;
        assert(!context.d_resonstruct[level]);
        context.d_resonstruct[level] = d_temp;
      } else {
        d_temp = d_output;
        assert(highSize == imageSize * context.batchSize);
      }
    } else {
      d_temp = (level > 0) ? context.d_resonstruct[level] : d_output;
    }
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedReconstructKernelRGB<T, F_T><<<grid, block, 0, stream>>>(
        d_reconstruct,
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        d_temp,
        context.batchSize);
    CUDA_CHECK(cudaGetLastError());
    d_reconstruct = d_temp;
  }
  assert(d_reconstruct == d_output);
  context.initialized = true;

  return cudaSuccess;
}

//------------------------------------------------------------------------------
// Explicit template instantiations for supported data types.
//------------------------------------------------------------------------------

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(T)  \
  template cudaError_t cudaBatchedLaplacianBlend<T>( \
      const T* h_image1,                             \
      const T* h_image2,                             \
      const T* h_mask,                               \
      T* h_output,                                   \
      int imageWidth,                                \
      int imageHeight,                               \
      int numLevels,                                 \
      int batchSize,                                 \
      cudaStream_t stream);

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(T) \
  template cudaError_t cudaBatchedLaplacianBlendWithContext<T>(  \
      const T* d_image1,                                         \
      const T* d_image2,                                         \
      const T* d_mask,                                           \
      T* d_output,                                               \
      CudaBatchLaplacianBlendContext<T>& context,                \
      cudaStream_t stream);

INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(float)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(float)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(unsigned char)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(unsigned char)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(__half)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(__half)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(__nv_bfloat16)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(__nv_bfloat16)
