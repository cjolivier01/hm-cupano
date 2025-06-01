// cudaBlend3.cu
#include "cudaBlend3.h"
#include "cudaTypes.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#define PRINT_STRANGE_ALPHAS
namespace {
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

#define T_CONST(_v$) static_cast<T>(_v$)
#define F_T_CONST(_v$) static_cast<F_T>(_v$)

// =============================================================================
// Templated CUDA Kernels for Batched Laplacian Blending of THREE images
// =============================================================================

// -----------------------------------------------------------------------------
// Fused downsample kernel for THREE images (RGB or RGBA).
// Each output pixel is computed by averaging a 2×2 block in each of the three inputs.
// "channels" is either 3 (RGB) or 4 (RGBA). Alpha channel is max-pooled.
template <typename T, typename F_T = float>
__global__ void FusedBatchedDownsampleKernel3(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    const T* __restrict__ input3,
    int inWidth,
    int inHeight,
    T* __restrict__ output1,
    T* __restrict__ output2,
    T* __restrict__ output3,
    int outWidth,
    int outHeight,
    int batchSize,
    int channels) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inImageSize = inWidth * inHeight * channels;
  int outImageSize = outWidth * outHeight * channels;
  const T* inImage1 = input1 + b * inImageSize;
  const T* inImage2 = input2 + b * inImageSize;
  const T* inImage3 = input3 + b * inImageSize;
  T* outImage1 = output1 + b * outImageSize;
  T* outImage2 = output2 + b * outImageSize;
  T* outImage3 = output3 + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;
  assert(channels <= 4);
  F_T sums1[3] = {0, 0, 0};
  F_T sums2[3] = {0, 0, 0};
  F_T sums3[3] = {0, 0, 0};
  T alpha1 = 0, alpha2 = 0, alpha3 = 0;
  const int sum_channels = std::min(channels, 3);
  int count = 0;

  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * channels;
        for (int c = 0; c < sum_channels; c++) {
          sums1[c] += static_cast<F_T>(inImage1[idx + c]);
          sums2[c] += static_cast<F_T>(inImage2[idx + c]);
          sums3[c] += static_cast<F_T>(inImage3[idx + c]);
        }
        if (channels == 4) {
          alpha1 = max(alpha1, inImage1[idx + 3]);
          alpha2 = max(alpha2, inImage2[idx + 3]);
          alpha3 = max(alpha3, inImage3[idx + 3]);
        }
        count++;
      }
    }
  }

  int outIdx = (y * outWidth + x) * channels;
  for (int c = 0; c < sum_channels; c++) {
    outImage1[outIdx + c] = static_cast<T>(sums1[c] / count);
    outImage2[outIdx + c] = static_cast<T>(sums2[c] / count);
    outImage3[outIdx + c] = static_cast<T>(sums3[c] / count);
  }
  if (channels == 4) {
    outImage1[outIdx + 3] = alpha1;
    outImage2[outIdx + 3] = alpha2;
    outImage3[outIdx + 3] = alpha3;
  }

#ifdef PRINT_STRANGE_ALPHAS
  if ((alpha1 != T_CONST(0) && alpha1 != T_CONST(255)) || (alpha2 != T_CONST(0) && alpha2 != T_CONST(255)) ||
      (alpha3 != T_CONST(0) && alpha3 != T_CONST(255))) {
    printf("FusedBatchedDownsampleKernel3(): Strange alphas %f, %f, %f\n", (float)alpha1, (float)alpha2, (float)alpha3);
  }
#endif
}

// -----------------------------------------------------------------------------
// Fused downsample kernel for a **3-channel mask**.
// Each output mask-pixel is the per-channel average over the 2×2 block.
template <typename T>
__global__ void FusedBatchedDownsampleMask3(
    const T* __restrict__ input, // [H×W×3], single (non-batched) mask
    int inWidth,
    int inHeight,
    T* __restrict__ output, // [outH×outW×3]
    int outWidth,
    int outHeight) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  int inX = x * 2;
  int inY = y * 2;
  // F_T_CONST(0.0f); // not used directly, but for casting
  float sums[3] = {0.0f, 0.0f, 0.0f};
  int count = 0;

  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * 3;
        for (int c = 0; c < 3; c++) {
          sums[c] += static_cast<float>(input[idx + c]);
        }
        count++;
      }
    }
  }

  int outIdx = (y * outWidth + x) * 3;
  for (int c = 0; c < 3; c++) {
    output[outIdx + c] = static_cast<T>(sums[c] / count);
  }
}

// -----------------------------------------------------------------------------
// Batched computation of the Laplacian for **one** image (RGB or RGBA).
// We will launch this separately for each of the three Gaussian pyramids.
template <typename T, typename F_T>
__global__ void BatchedComputeLaplacianKernel(
    const T* __restrict__ gaussHigh,
    int highWidth,
    int highHeight,
    const T* __restrict__ gaussLow,
    int lowWidth,
    int lowHeight,
    T* __restrict__ laplacian,
    int batchSize,
    int channels) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int highImageSize = highWidth * highHeight * channels;
  int lowImageSize = lowWidth * lowHeight * channels;
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

  int idxHigh = (y * highWidth + x) * channels;
  for (int c = 0; c < channels; c++) {
    if (channels == 4 && c == 3) {
      // For the alpha channel, copy the high-res value.
      lapImage[idxHigh + c] = highImage[idxHigh + c];
    } else {
      int idx00 = (gyi * lowWidth + gxi) * channels + c;
      int idx10 = (gyi * lowWidth + gxi1) * channels + c;
      int idx01 = (gyi1 * lowWidth + gxi) * channels + c;
      int idx11 = (gyi1 * lowWidth + gxi1) * channels + c;
      F_T val00 = static_cast<F_T>(lowImage[idx00]);
      F_T val10 = static_cast<F_T>(lowImage[idx10]);
      F_T val01 = static_cast<F_T>(lowImage[idx01]);
      F_T val11 = static_cast<F_T>(lowImage[idx11]);

      // bilinear interpolation from low to high
      F_T val0 = val00 * (1.0f - dx) + val10 * dx;
      F_T val1 = val01 * (1.0f - dx) + val11 * dx;
      F_T upVal = val0 * (1.0f - dy) + val1 * dy;

      lapImage[idxHigh + c] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + c]) - upVal);
    }
  }
}

// -----------------------------------------------------------------------------
// Batched blend kernel for THREE images.
// For each channel, the three Laplacian pyramids are blended with a weighted average
// using a 3-channel mask (m₁, m₂, m₃). If channels==4, the alpha channel is also
// blended by weighted average of alpha1, alpha2, alpha3.
template <typename T, typename F_T>
__global__ void BatchedBlendKernel3(
    const T* __restrict__ lap1,
    const T* __restrict__ lap2,
    const T* __restrict__ lap3,
    const T* __restrict__ mask, // [H×W×3], single (non-batched) mask
    T* __restrict__ blended,
    int width,
    int height,
    int batchSize,
    int channels) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height)
    return;

  int imageSize = width * height * channels;
  const T* lap1Image = lap1 + b * imageSize;
  const T* lap2Image = lap2 + b * imageSize;
  const T* lap3Image = lap3 + b * imageSize;
  const T* maskImage = mask + (y * width + x) * 3; // 3 weights per pixel
  T* blendImage = blended + b * imageSize;

  // Fetch mask weights, cast to F_T
  F_T m1 = static_cast<F_T>(maskImage[0]);
  F_T m2 = static_cast<F_T>(maskImage[1]);
  F_T m3 = static_cast<F_T>(maskImage[2]);

  // printf("d=%p, b=%d, m1=%f, m2=%f, m3=%f\n", blended, (int)b, (float)m1, (float)m2, (float)m3);

  int idx = (y * width + x) * channels;
  for (int c = 0; c < channels; c++) {
    F_T v1 = static_cast<F_T>(lap1Image[idx + c]);
    F_T v2 = static_cast<F_T>(lap2Image[idx + c]);
    F_T v3 = static_cast<F_T>(lap3Image[idx + c]);
    F_T blendedVal = m1 * v1 + m2 * v2 + m3 * v3;
    // printf("pos=%d, pix1=%f, pix2=%f, pix3=%f -> blended=%f\n", (int)idx + c, (float)v1, (float)v2, (float)v3, (float)blendedVal);
    blendImage[idx + c] = static_cast<T>(blendedVal);
  }
}

// -----------------------------------------------------------------------------
// Batched reconstruction kernel (same as two-image version).
// Reconstructs the high-res image from a lower-res blended pyramid and adds the Laplacian.
template <typename T, typename F_T>
__global__ void BatchedReconstructKernel(
    const T* __restrict__ lowerRes,
    int lowWidth,
    int lowHeight,
    const T* __restrict__ lap,
    int highWidth,
    int highHeight,
    T* __restrict__ reconstruction,
    int batchSize,
    int channels) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highWidth || y >= highHeight)
    return;

  int lowImageSize = lowWidth * lowHeight * channels;
  int highImageSize = highWidth * highHeight * channels;
  const T* lowImage = lowerRes + b * lowImageSize;
  const T* lapImage = lap + b * highImageSize;
  T* reconImage = reconstruction + b * highImageSize;

  const F_T F_ONE = static_cast<F_T>(1.0f);
  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idxOut = (y * highWidth + x) * channels;
  for (int c = 0; c < channels; c++) {
    if (channels == 4 && c == 3) {
      // Alpha: copy from Laplacian’s alpha directly (already blended).
      reconImage[idxOut + 3] = lapImage[idxOut + 3];
#ifdef PRINT_STRANGE_ALPHAS
      float alpha = static_cast<float>(reconImage[idxOut + 3]);
      if (alpha != 0 && alpha != 255) {
        printf("BatchedReconstructKernel(): Strange alpha %f\n", alpha);
      }
#endif
    } else {
      int idx00 = (gyi * lowWidth + gxi) * channels + c;
      int idx10 = (gyi * lowWidth + gxi1) * channels + c;
      int idx01 = (gyi1 * lowWidth + gxi) * channels + c;
      int idx11 = (gyi1 * lowWidth + gxi1) * channels + c;
      F_T val00 = static_cast<F_T>(lowImage[idx00]);
      F_T val10 = static_cast<F_T>(lowImage[idx10]);
      F_T val01 = static_cast<F_T>(lowImage[idx01]);
      F_T val11 = static_cast<F_T>(lowImage[idx11]);

      // bilinear upsampling
      F_T upVal = (val00 * (F_ONE - dx) + val10 * dx) * (F_ONE - dy) + (val01 * (F_ONE - dx) + val11 * dx) * dy;
      reconImage[idxOut + c] = static_cast<T>(upVal + static_cast<F_T>(lapImage[idxOut + c]));
    }
  }
}
} // namespace

// =============================================================================
// Templated Host Functions: Batched Laplacian Blending for THREE images
// =============================================================================

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlend3(
    const T* h_image1,
    const T* h_image2,
    const T* h_image3,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels, // New parameter: 3 for RGB or 4 for RGBA
    int numLevels,
    int batchSize,
    cudaStream_t stream) {
  // Size computations
  size_t imageSize = static_cast<size_t>(imageWidth) * imageHeight * channels * sizeof(T);
  // size_t maskSizeLevel = static_cast<size_t>(imageWidth) * imageHeight * 3 * sizeof(T); // 3-channel mask

  // Storage for device pointers per level:
  std::vector<T*> d_gauss1(numLevels), d_gauss2(numLevels), d_gauss3(numLevels);
  std::vector<T*> d_maskPyr(numLevels);
  std::vector<T*> d_lap1(numLevels), d_lap2(numLevels), d_lap3(numLevels);
  std::vector<T*> d_blend(numLevels);

  // Compute widths/heights of each pyramid level:
  std::vector<int> widths(numLevels), heights(numLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int i = 1; i < numLevels; i++) {
    widths[i] = (widths[i - 1] + 1) / 2;
    heights[i] = (heights[i - 1] + 1) / 2;
  }

  // --------------- Allocate level-0 arrays and copy input data to device ---------------
  size_t sizeImgLevel0 = static_cast<size_t>(widths[0]) * heights[0] * channels * batchSize * sizeof(T);
  size_t sizeMaskLevel0 = static_cast<size_t>(widths[0]) * heights[0] * 3 * sizeof(T);
  CUDA_CHECK(cudaMalloc((void**)&d_gauss1[0], sizeImgLevel0));
  CUDA_CHECK(cudaMalloc((void**)&d_gauss2[0], sizeImgLevel0));
  CUDA_CHECK(cudaMalloc((void**)&d_gauss3[0], sizeImgLevel0));
  CUDA_CHECK(cudaMalloc((void**)&d_maskPyr[0], sizeMaskLevel0));

  CUDA_CHECK(cudaMemcpyAsync(d_gauss1[0], h_image1, imageSize * batchSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_gauss2[0], h_image2, imageSize * batchSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_gauss3[0], h_image3, imageSize * batchSize, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_maskPyr[0], h_mask, sizeMaskLevel0, cudaMemcpyHostToDevice, stream));

  // --------------- Allocate device memory for higher pyramid levels (images + mask) ---------------
  for (int level = 1; level < numLevels; level++) {
    size_t sizeImg = static_cast<size_t>(widths[level]) * heights[level] * channels * batchSize * sizeof(T);
    size_t sizeMask = static_cast<size_t>(widths[level]) * heights[level] * 3 * sizeof(T);
    CUDA_CHECK(cudaMalloc((void**)&d_gauss1[level], sizeImg));
    CUDA_CHECK(cudaMalloc((void**)&d_gauss2[level], sizeImg));
    CUDA_CHECK(cudaMalloc((void**)&d_gauss3[level], sizeImg));
    CUDA_CHECK(cudaMalloc((void**)&d_maskPyr[level], sizeMask));
  }

  // --------------- Build Gaussian pyramids (downsample) ---------------
  dim3 block(16, 16, 1);
  for (int level = 0; level < numLevels - 1; level++) {
    int inW = widths[level];
    int inH = heights[level];
    int outW = widths[level + 1];
    int outH = heights[level + 1];

    dim3 gridImg((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, batchSize);

    // Downsample the three image sets
    FusedBatchedDownsampleKernel3<T><<<gridImg, block, 0, stream>>>(
        d_gauss1[level],
        d_gauss2[level],
        d_gauss3[level],
        inW,
        inH,
        d_gauss1[level + 1],
        d_gauss2[level + 1],
        d_gauss3[level + 1],
        outW,
        outH,
        batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());

    // Downsample the 3-channel mask (non-batched)
    dim3 gridMask((outW + block.x - 1) / block.x, (outH + block.y - 1) / block.y, 1);

    FusedBatchedDownsampleMask3<T>
        <<<gridMask, block, 0, stream>>>(d_maskPyr[level], inW, inH, d_maskPyr[level + 1], outW, outH);
    CUDA_CHECK(cudaGetLastError());
  }

  // --------------- Build Laplacian pyramids ---------------
  for (int level = 0; level < numLevels; level++) {
    size_t sizeImg = static_cast<size_t>(widths[level]) * heights[level] * channels * batchSize * sizeof(T);
    CUDA_CHECK(cudaMalloc((void**)&d_lap1[level], sizeImg));
    CUDA_CHECK(cudaMalloc((void**)&d_lap2[level], sizeImg));
    CUDA_CHECK(cudaMalloc((void**)&d_lap3[level], sizeImg));
  }

  // For levels 0..numLevels-2, compute laplacian = gaussHigh – upsample(gaussLow)
  for (int level = 0; level < numLevels - 1; level++) {
    int wH = widths[level];
    int hH = heights[level];
    int wL = widths[level + 1];
    int hL = heights[level + 1];
    dim3 gridLap((wH + block.x - 1) / block.x, (hH + block.y - 1) / block.y, batchSize);

    // Laplacian for image 1
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        d_gauss1[level], wH, hH, d_gauss1[level + 1], wL, hL, d_lap1[level], batchSize, channels);
    CUDA_CHECK(cudaGetLastError());

    // Laplacian for image 2
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        d_gauss2[level], wH, hH, d_gauss2[level + 1], wL, hL, d_lap2[level], batchSize, channels);
    CUDA_CHECK(cudaGetLastError());

    // Laplacian for image 3
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        d_gauss3[level], wH, hH, d_gauss3[level + 1], wL, hL, d_lap3[level], batchSize, channels);
    CUDA_CHECK(cudaGetLastError());
  }

  // For the last (coarsest) level, copy Gaussian → Laplacian directly
  int last = numLevels - 1;
  size_t lastSize = static_cast<size_t>(widths[last]) * heights[last] * channels * batchSize * sizeof(T);
  CUDA_CHECK(cudaMemcpyAsync(d_lap1[last], d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_lap2[last], d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_lap3[last], d_gauss3[last], lastSize, cudaMemcpyDeviceToDevice, stream));

  // --------------- Blend the Laplacian pyramids using the 3-channel mask ---------------
  for (int level = 0; level < numLevels; level++) {
    int w = widths[level];
    int h = heights[level];
    size_t sizeImg = static_cast<size_t>(w) * h * channels * batchSize * sizeof(T);
    CUDA_CHECK(cudaMalloc((void**)&d_blend[level], sizeImg));

    dim3 gridBlend((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, batchSize);

    BatchedBlendKernel3<T, F_T><<<gridBlend, block, 0, stream>>>(
        d_lap1[level],
        d_lap2[level],
        d_lap3[level],
        d_maskPyr[level], // 3-channel mask at this level
        d_blend[level],
        w,
        h,
        batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
  }

  // --------------- Reconstruct the final blended image (bottom-up) ---------------
  T* d_reconstruct = nullptr;
  size_t sizeCoarse = static_cast<size_t>(widths[last]) * heights[last] * channels * batchSize * sizeof(T);
  CUDA_CHECK(cudaMalloc((void**)&d_reconstruct, sizeCoarse));
  CUDA_CHECK(cudaMemcpyAsync(d_reconstruct, d_blend[last], sizeCoarse, cudaMemcpyDeviceToDevice, stream));

  for (int level = numLevels - 2; level >= 0; level--) {
    int wH = widths[level];
    int hH = heights[level];
    int wL = widths[level + 1];
    int hL = heights[level + 1];
    size_t sizeHigh = static_cast<size_t>(wH) * hH * channels * batchSize * sizeof(T);
    T* d_temp = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_temp, sizeHigh));

    dim3 gridRecon((wH + block.x - 1) / block.x, (hH + block.y - 1) / block.y, batchSize);

    BatchedReconstructKernel<T, F_T>
        <<<gridRecon, block, 0, stream>>>(d_reconstruct, wL, hL, d_blend[level], wH, hH, d_temp, batchSize, channels);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_reconstruct);
    d_reconstruct = d_temp;
  }

  // Copy final blended image(s) back to host
  CUDA_CHECK(cudaMemcpyAsync(h_output, d_reconstruct, imageSize * batchSize, cudaMemcpyDeviceToHost, stream));
  cudaFree(d_reconstruct);

  // --------------- Cleanup allocated device memory ---------------
  for (int level = 0; level < numLevels; level++) {
    cudaFree(d_gauss1[level]);
    cudaFree(d_gauss2[level]);
    cudaFree(d_gauss3[level]);
    cudaFree(d_maskPyr[level]);
    cudaFree(d_lap1[level]);
    cudaFree(d_lap2[level]);
    cudaFree(d_lap3[level]);
    cudaFree(d_blend[level]);
  }

  return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Templated version with context, now accepting three images and a 3-channel mask.
template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendWithContext3(
    const T* d_image1,
    const T* d_image2,
    const T* d_image3,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext3<T>& context,
    int channels,
    cudaStream_t stream) {
  // --------------- Initialization: allocate buffers if needed ---------------
  if (!context.initialized) {
    // Set up pyramid dimensions
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < context.numLevels; i++) {
      context.widths[i] = (context.widths[i - 1] + 1) / 2;
      context.heights[i] = (context.heights[i - 1] + 1) / 2;
      assert(context.widths[i] && context.heights[i]);
    }

    for (int level = 0; level < context.numLevels; level++) {
      int w = context.widths[level];
      int h = context.heights[level];
      size_t sizeImg = static_cast<size_t>(w) * h * channels * context.batchSize * sizeof(T);
      size_t sizeMask = static_cast<size_t>(w) * h * 3 * sizeof(T);

      // Allocate Laplacian and blend buffers
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap1[level], sizeImg));
      context.allocation_size += sizeImg;
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap2[level], sizeImg));
      context.allocation_size += sizeImg;
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap3[level], sizeImg));
      context.allocation_size += sizeImg;
      CUDA_CHECK(cudaMalloc((void**)&context.d_blend[level], sizeImg));
      context.allocation_size += sizeImg;

      if (level > 0) {
        // For levels > 0, allocate gauss and mask as well
        CUDA_CHECK(cudaMalloc((void**)&context.d_maskPyr[level], sizeMask));
        context.allocation_size += sizeMask;
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss1[level], sizeImg));
        context.allocation_size += sizeImg;
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss2[level], sizeImg));
        context.allocation_size += sizeImg;
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss3[level], sizeImg));
        context.allocation_size += sizeImg;
        CUDA_CHECK(cudaMalloc((void**)&context.d_reconstruct[level], sizeImg));
        context.allocation_size += sizeImg;
      } else {
        // Level 0: pointers come from user
        context.d_maskPyr[0] = const_cast<T*>(d_mask);
        context.d_gauss1[0] = const_cast<T*>(d_image1);
        context.d_gauss2[0] = const_cast<T*>(d_image2);
        context.d_gauss3[0] = const_cast<T*>(d_image3);
        // The output pointer will be used for reconstruction when level==0
        context.d_reconstruct[0] = d_output;
      }
    }
  }

  dim3 block(16, 16, 1);

  // --------------- Build Gaussian pyramids (downsample) ---------------
  for (int level = 0; level < context.numLevels - 1; level++) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    dim3 gridImg((wL + block.x - 1) / block.x, (hL + block.y - 1) / block.y, context.batchSize);

    // Downsample the three image sets
    FusedBatchedDownsampleKernel3<T><<<gridImg, block, 0, stream>>>(
        context.d_gauss1[level],
        context.d_gauss2[level],
        context.d_gauss3[level],
        wH,
        hH,
        context.d_gauss1[level + 1],
        context.d_gauss2[level + 1],
        context.d_gauss3[level + 1],
        wL,
        hL,
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());

    // Downsample the 3-channel mask
    dim3 gridMask((wL + block.x - 1) / block.x, (hL + block.y - 1) / block.y, 1);

    FusedBatchedDownsampleMask3<T>
        <<<gridMask, block, 0, stream>>>(context.d_maskPyr[level], wH, hH, context.d_maskPyr[level + 1], wL, hL);
    CUDA_CHECK(cudaGetLastError());
  }

  // --------------- Build Laplacian pyramids ---------------
  for (int level = 0; level < context.numLevels - 1; level++) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    dim3 gridLap((wH + block.x - 1) / block.x, (hH + block.y - 1) / block.y, context.batchSize);

    // Laplacian for image 1
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        context.d_gauss1[level],
        wH,
        hH,
        context.d_gauss1[level + 1],
        wL,
        hL,
        context.d_lap1[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());

    // Laplacian for image 2
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        context.d_gauss2[level],
        wH,
        hH,
        context.d_gauss2[level + 1],
        wL,
        hL,
        context.d_lap2[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());

    // Laplacian for image 3
    BatchedComputeLaplacianKernel<T, F_T><<<gridLap, block, 0, stream>>>(
        context.d_gauss3[level],
        wH,
        hH,
        context.d_gauss3[level + 1],
        wL,
        hL,
        context.d_lap3[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
  }

  // Coarsest level copy (Gaussian → Laplacian)
  int last = context.numLevels - 1;
  size_t lastSize =
      static_cast<size_t>(context.widths[last]) * context.heights[last] * channels * context.batchSize * sizeof(T);

  CUDA_CHECK(cudaMemcpyAsync(context.d_lap1[last], context.d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(context.d_lap2[last], context.d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(context.d_lap3[last], context.d_gauss3[last], lastSize, cudaMemcpyDeviceToDevice, stream));

  // --------------- Blend the Laplacian pyramids ---------------
  for (int level = 0; level < context.numLevels; level++) {
    int w = context.widths[level];
    int h = context.heights[level];
    dim3 gridBlend((w + block.x - 1) / block.x, (h + block.y - 1) / block.y, context.batchSize);

    BatchedBlendKernel3<T, F_T><<<gridBlend, block, 0, stream>>>(
        context.d_lap1[level],
        context.d_lap2[level],
        context.d_lap3[level],
        context.d_maskPyr[level],
        context.d_blend[level],
        w,
        h,
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
  }

  // --------------- Reconstruct the final image ---------------
  T* d_reconstruct = nullptr;
  if (!context.initialized) {
    // Coarsest level allocation
    if (context.numLevels > 1) {
      size_t sizeCoarse =
          static_cast<size_t>(context.widths[last]) * context.heights[last] * channels * context.batchSize * sizeof(T);
      CUDA_CHECK(cudaMalloc((void**)&d_reconstruct, sizeCoarse));
      context.allocation_size += sizeCoarse;
      context.d_reconstruct[last] = d_reconstruct;
    } else {
      // If only one level, reconstruct directly into d_output
      d_reconstruct = d_output;
      context.d_reconstruct[last] = d_reconstruct;
    }
  } else {
    // Already allocated in a previous call
    d_reconstruct = context.d_reconstruct[last];
    assert(d_reconstruct);
  }

  // Copy blended at coarsest level into d_reconstruct
  CUDA_CHECK(cudaMemcpyAsync(
      d_reconstruct,
      context.d_blend[last],
      static_cast<size_t>(context.widths[last]) * context.heights[last] * channels * context.batchSize * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream));

  for (int level = context.numLevels - 2; level >= 0; level--) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    T* d_temp = nullptr;
    if (!context.initialized) {
      if (level > 0) {
        size_t sizeHigh = static_cast<size_t>(wH) * hH * channels * context.batchSize * sizeof(T);
        CUDA_CHECK(cudaMalloc((void**)&d_temp, sizeHigh));
        context.allocation_size += sizeHigh;
        context.d_reconstruct[level] = d_temp;
      } else {
        // Top level: write directly to d_output
        d_temp = d_output;
      }
    } else {
      d_temp = (level > 0) ? context.d_reconstruct[level] : d_output;
    }

    dim3 gridRecon((wH + block.x - 1) / block.x, (hH + block.y - 1) / block.y, context.batchSize);

    BatchedReconstructKernel<T, F_T><<<gridRecon, block, 0, stream>>>(
        d_reconstruct, wL, hL, context.d_blend[level], wH, hH, d_temp, context.batchSize, channels);
    CUDA_CHECK(cudaGetLastError());

    d_reconstruct = d_temp;
  }

  context.initialized = true;
  return cudaSuccess;
}

// =============================================================================
// Explicit template instantiations (already declared in header).
// =============================================================================

template cudaError_t cudaBatchedLaplacianBlend3<float, float>(
    const float* h_image1,
    const float* h_image2,
    const float* h_image3,
    const float* h_mask,
    float* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int numLevels,
    int batchSize,
    cudaStream_t stream);

template cudaError_t cudaBatchedLaplacianBlend3<unsigned char, float>(
    const unsigned char* h_image1,
    const unsigned char* h_image2,
    const unsigned char* h_image3,
    const unsigned char* h_mask,
    unsigned char* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int numLevels,
    int batchSize,
    cudaStream_t stream);

template cudaError_t cudaBatchedLaplacianBlendWithContext3<float, float>(
    const float* d_image1,
    const float* d_image2,
    const float* d_image3,
    const float* d_mask,
    float* d_output,
    CudaBatchLaplacianBlendContext3<float>& context,
    int channels,
    cudaStream_t stream);

template cudaError_t cudaBatchedLaplacianBlendWithContext3<unsigned char, float>(
    const unsigned char* d_image1,
    const unsigned char* d_image2,
    const unsigned char* d_image3,
    const unsigned char* d_mask,
    unsigned char* d_output,
    CudaBatchLaplacianBlendContext3<unsigned char>& context,
    int channels,
    cudaStream_t stream);
