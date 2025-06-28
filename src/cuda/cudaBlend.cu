#include "cudaBlend.h"
#include "cudaUtils.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#define PRINT_STRANGE_ALPHAS

#define EXTRA_ALPHA_CHECKS

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
// Templated CUDA Kernels for Batched Laplacian Blending (now with a "channels" parameter)
// =============================================================================

// -----------------------------------------------------------------------------
// Fused downsample kernel for two images.
// "channels" is either 3 (RGB) or 4 (RGBA). Each output pixel is computed by averaging a 2x2 block.
#ifdef EXTRA_ALPHA_CHECKS
// Fused downsample kernel for TWO images (RGB or RGBA).
// Each output pixel is the average of a 2×2 block in each input.
// If channels==4, any pixel with alpha==0 is omitted from its image’s average.
// Alpha is still max‐pooled.
template <typename T, typename F_T = float>
__global__ void FusedBatchedDownsampleKernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int inWidth,
    int inHeight,
    T* __restrict__ output1,
    T* __restrict__ output2,
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

  const int inImageSize = inWidth * inHeight * channels;
  const int outImageSize = outWidth * outHeight * channels;
  const T* in1 = input1 + b * inImageSize;
  const T* in2 = input2 + b * inImageSize;
  T* out1 = output1 + b * outImageSize;
  T* out2 = output2 + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;

  const int sumCh = std::min(channels, 3);

  // accumulators and counters for each image
  F_T sums1[3] = {0}, sums2[3] = {0};
  int count1 = 0, count2 = 0;
  // max-pooled alpha
  T alpha1 = 0, alpha2 = 0;

  // loop over the 2×2 block
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      int ix = inX + dx, iy = inY + dy;
      if (ix >= inWidth || iy >= inHeight)
        continue;

      int idx = (iy * inWidth + ix) * channels;

      // IMAGE 1
      bool keep1 = true;
      if (channels == 4) {
        T a1 = in1[idx + 3];
        alpha1 = max_of(alpha1, a1);
        keep1 = !is_zero(a1);
      }
      if (keep1) {
        for (int c = 0; c < sumCh; ++c)
          sums1[c] += static_cast<F_T>(in1[idx + c]);
        ++count1;
      }

      // IMAGE 2
      bool keep2 = true;
      if (channels == 4) {
        T a2 = in2[idx + 3];
        alpha2 = max_of(alpha2, a2);
        keep2 = !is_zero(a2);
      }
      if (keep2) {
        for (int c = 0; c < sumCh; ++c)
          sums2[c] += static_cast<F_T>(in2[idx + c]);
        ++count2;
      }
    }
  }

  int outIdx = (y * outWidth + x) * channels;

  // write RGB averages (or zero if no contributors)
  for (int c = 0; c < sumCh; ++c) {
    out1[outIdx + c] = count1 > 0 ? static_cast<T>(sums1[c] / count1) : T(0);
    out2[outIdx + c] = count2 > 0 ? static_cast<T>(sums2[c] / count2) : T(0);
  }

  // write alpha as max-pooled
  if (channels == 4) {
    out1[outIdx + 3] = alpha1;
    out2[outIdx + 3] = alpha2;
  }

#ifdef PRINT_STRANGE_ALPHAS
  // stupid overload issues
  if constexpr (!std::is_same<T, __half>::value) {
    if ((alpha1 != T_CONST(0) && alpha1 != T_CONST(255)) || (alpha2 != T_CONST(0) && alpha2 != T_CONST(255))) {
      printf("FusedBatchedDownsampleKernel(): Strange alphas %f and %f\n", (float)alpha1, (float)alpha2);
    }
  }
#endif
}
#else
template <typename T, typename F_T = float>
__global__ void FusedBatchedDownsampleKernel(
    const T* __restrict__ input1,
    const T* __restrict__ input2,
    int inWidth,
    int inHeight,
    T* __restrict__ output1,
    T* __restrict__ output2,
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
  T* outImage1 = output1 + b * outImageSize;
  T* outImage2 = output2 + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;
  assert(channels <= 4);
  F_T sums1[3] = {0, 0, 0};
  F_T sums2[3] = {0, 0, 0};
  T alpha1 = 0;
  T alpha2 = 0;
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
        }
        if (channels == 4) {
          alpha1 = std::max(alpha1, inImage1[idx + 3]);
          alpha2 = std::max(alpha2, inImage2[idx + 3]);
        }
        count++;
      }
    }
  }
  int outIdx = (y * outWidth + x) * channels;
  for (int c = 0; c < sum_channels; c++) {
    outImage1[outIdx + c] = static_cast<T>(sums1[c] / count);
    outImage2[outIdx + c] = static_cast<T>(sums2[c] / count);
  }
  if (channels == 4) {
    outImage1[outIdx + 3] = alpha1;
    outImage2[outIdx + 3] = alpha2;
  }
#ifdef PRINT_STRANGE_ALPHAS
  if ((alpha1 != T_CONST(0) && alpha1 != T_CONST(255)) || (alpha2 != T_CONST(0) && alpha2 != T_CONST(255))) {
    printf("FusedBatchedDownsampleKernel(): Strange alphas %f and %f\n", (float)alpha1, (float)alpha2);
  }
#endif
}
#endif
// -----------------------------------------------------------------------------
// Batched downsample kernel for a single image (RGB or RGBA).
// template <typename T>
// __global__ void BatchedDownsampleKernel(
//     const T* __restrict__ input,
//     int inWidth,
//     int inHeight,
//     T* __restrict__ output,
//     int outWidth,
//     int outHeight,
//     int batchSize,
//     int channels) {
//   int b = blockIdx.z;
//   if (b >= batchSize)
//     return;

//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x >= outWidth || y >= outHeight)
//     return;

//   int inImageSize = inWidth * inHeight * channels;
//   int outImageSize = outWidth * outHeight * channels;
//   const T* inImage = input + b * inImageSize;
//   T* outImage = output + b * outImageSize;

//   int inX = x * 2;
//   int inY = y * 2;
//   float sums[4] = {0, 0, 0, 0};
//   int count = 0;
//   for (int dy = 0; dy < 2; dy++) {
//     for (int dx = 0; dx < 2; dx++) {
//       int ix = inX + dx;
//       int iy = inY + dy;
//       if (ix < inWidth && iy < inHeight) {
//         int idx = (iy * inWidth + ix) * channels;
//         for (int c = 0; c < channels; c++) {
//           sums[c] += static_cast<float>(inImage[idx + c]);
//         }
//         count++;
//       }
//     }
//   }
//   int outIdx = (y * outWidth + x) * channels;
//   for (int c = 0; c < channels; c++) {
//     outImage[outIdx + c] = static_cast<T>(sums[c] / count);
//   }
// }

// -----------------------------------------------------------------------------
// Batched downsample kernel for a single-channel mask (unchanged).
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
  for (int dy = 0; dy < 2; dy++) {
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
// Batched upsample kernel for images (RGB or RGBA) using bilinear interpolation.
// template <typename T, typename F_T>
// __global__ void BatchedUpsampleKernel(
//     const T* __restrict__ input,
//     int inWidth,
//     int inHeight,
//     T* __restrict__ output,
//     int outWidth,
//     int outHeight,
//     int batchSize,
//     int channels) {
//   int b = blockIdx.z;
//   if (b >= batchSize)
//     return;

//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x >= outWidth || y >= outHeight)
//     return;

//   int inImageSize = inWidth * inHeight * channels;
//   int outImageSize = outWidth * outHeight * channels;
//   const T* inImage = input + b * inImageSize;
//   T* outImage = output + b * outImageSize;

//   F_T gx = static_cast<F_T>(x) / 2.0f;
//   F_T gy = static_cast<F_T>(y) / 2.0f;
//   int gxi = floorf(gx);
//   int gyi = floorf(gy);
//   F_T dx = gx - gxi;
//   F_T dy = gy - gyi;
//   int gxi1 = min(gxi + 1, inWidth - 1);
//   int gyi1 = min(gyi + 1, inHeight - 1);

//   int idx00 = (gyi * inWidth + gxi) * channels;
//   int idx10 = (gyi * inWidth + gxi1) * channels;
//   int idx01 = (gyi1 * inWidth + gxi) * channels;
//   int idx11 = (gyi1 * inWidth + gxi1) * channels;

//   int idxOut = (y * outWidth + x) * channels;
//   for (int c = 0; c < channels; c++) {
//     F_T val00 = static_cast<F_T>(inImage[idx00 + c]);
//     F_T val10 = static_cast<F_T>(inImage[idx10 + c]);
//     F_T val01 = static_cast<F_T>(inImage[idx01 + c]);
//     F_T val11 = static_cast<F_T>(inImage[idx11 + c]);
//     F_T val0 = val00 * (1.0f - dx) + val10 * dx;
//     F_T val1 = val01 * (1.0f - dx) + val11 * dx;
//     F_T outVal = val0 * (1.0f - dy) + val1 * dy;
//     outImage[idxOut + c] = static_cast<T>(outVal);
//   }
// }

// -----------------------------------------------------------------------------
// Batched computation of the Laplacian.
// For each channel, compute Laplacian = gaussHigh - upsample(gaussLow).
// For RGBA images (channels==4), the alpha channel (c==3) is simply copied.
#ifdef EXTRA_ALPHA_CHECKS
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

  // map (x,y) in high-res to fractional coord in low-res
  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;
  int gxi = floorf(gx);
  int gyi = floorf(gy);
  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  int idxHigh = (y * highWidth + x) * channels;

  for (int c = 0; c < channels; ++c) {
    if (channels == 4 && c == 3) {
      // alpha channel: just copy high-res alpha
      lapImage[idxHigh + c] = highImage[idxHigh + c];
    } else {
      // gather neighbor indices
      int base00 = (gyi * lowWidth + gxi) * channels;
      int base10 = (gyi * lowWidth + gxi1) * channels;
      int base01 = (gyi1 * lowWidth + gxi) * channels;
      int base11 = (gyi1 * lowWidth + gxi1) * channels;
      int idx00 = base00 + c;
      int idx10 = base10 + c;
      int idx01 = base01 + c;
      int idx11 = base11 + c;

      // sample values
      F_T v00 = static_cast<F_T>(lowImage[idx00]);
      F_T v10 = static_cast<F_T>(lowImage[idx10]);
      F_T v01 = static_cast<F_T>(lowImage[idx01]);
      F_T v11 = static_cast<F_T>(lowImage[idx11]);

      F_T upVal;
      if (channels == 4) {
        // compute bilinear weights
        F_T w00 = (1 - dx) * (1 - dy);
        F_T w10 = dx * (1 - dy);
        F_T w01 = (1 - dx) * dy;
        F_T w11 = dx * dy;

        // accumulate only non-transparent neighbors
        F_T sumW = 0, sumV = 0;
        // alpha offsets
        int aOff = 3;
        if (!is_zero(lowImage[base00 + aOff])) {
          sumW += w00;
          sumV += v00 * w00;
        }
        if (!is_zero(lowImage[base10 + aOff])) {
          sumW += w10;
          sumV += v10 * w10;
        }
        if (!is_zero(lowImage[base01 + aOff])) {
          sumW += w01;
          sumV += v01 * w01;
        }
        if (!is_zero(lowImage[base11 + aOff])) {
          sumW += w11;
          sumV += v11 * w11;
        }

        upVal = (sumW > 0) ? (sumV / sumW) : F_T(0);
      } else {
        // standard bilinear
        F_T v0 = v00 * (1 - dx) + v10 * dx;
        F_T v1 = v01 * (1 - dx) + v11 * dx;
        upVal = v0 * (1 - dy) + v1 * dy;
      }

      // laplacian = high-res - upsampled
      lapImage[idxHigh + c] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + c]) - upVal);
    }
  }
}
#else
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
      // For the alpha channel, copy the high resolution value.
      lapImage[idxHigh + c] = highImage[idxHigh + c];
    } else {
      int idx00 = (gyi * lowWidth + gxi) * channels + c;
      int idx10 = (gyi * lowWidth + gxi1) * channels + c;
      int idx01 = (gyi1 * lowWidth + gxi) * channels + c;
      int idx11 = (gyi1 * lowWidth + gxi1) * channels + c;
      const F_T F_ONE = static_cast<F_T>(1.0);
      F_T val00 = static_cast<F_T>(lowImage[idx00]);
      F_T val10 = static_cast<F_T>(lowImage[idx10]);
      F_T val01 = static_cast<F_T>(lowImage[idx01]);
      F_T val11 = static_cast<F_T>(lowImage[idx11]);
      F_T val0 = val00 * (F_ONE - dx) + val10 * dx;
      F_T val1 = val01 * (F_ONE - dx) + val11 * dx;
      F_T upVal = val0 * (F_ONE - dy) + val1 * dy;
      lapImage[idxHigh + c] = static_cast<T>(static_cast<F_T>(highImage[idxHigh + c]) - upVal);
    }
  }
}
#endif
// -----------------------------------------------------------------------------
// Batched blend kernel for images.
// For each channel the two Laplacian pyramids are blended with a weighted average.
// (For alpha in RGBA, a simple weighted blend is performed.)
template <typename T, typename F_T>
__global__ void BatchedBlendKernel(
    const T* __restrict__ lap1,
    const T* __restrict__ lap2,
    const T* __restrict__ mask,
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
  // The mask is non-batched.
  const T* maskImage = mask;
  T* blendImage = blended + b * imageSize;

  constexpr const F_T F_ONE = static_cast<F_T>(1.0);
  int idx = (y * width + x) * channels;
  F_T m = static_cast<F_T>(maskImage[y * width + x]);
  F_T mm1 = F_ONE - m;
  int max_channel = std::min(channels, 3);
  if (channels == 4) {
    const T T_ZERO = static_cast<T>(0);
    const T alpha1 = lap1Image[idx + 3];
    const T alpha2 = lap2Image[idx + 3];
    if (is_zero(alpha1)) {
      for (int c = 0; c < channels; c++) {
        blendImage[idx + c] = static_cast<F_T>(lap2Image[idx + c]);
      }
    } else if (is_zero(alpha2)) {
      for (int c = 0; c < channels; c++) {
        blendImage[idx + c] = static_cast<F_T>(lap1Image[idx + c]);
      }
    } else {
      for (int c = 0; c < max_channel; c++) {
        blendImage[idx + c] =
            static_cast<T>(m * static_cast<F_T>(lap1Image[idx + c]) + mm1 * static_cast<F_T>(lap2Image[idx + c]));
      }
      // Both alphas not zero, we assume both are the same value, so copy one of them into dest
      blendImage[idx + 3] = alpha2;
    }
  } else {
    for (int c = 0; c < max_channel; c++) {
      blendImage[idx + c] =
          static_cast<T>(m * static_cast<F_T>(lap1Image[idx + c]) + mm1 * static_cast<F_T>(lap2Image[idx + c]));
    }
  }
}

// -----------------------------------------------------------------------------
// Batched reconstruction kernel for images.
// Reconstruction is performed per channel via bilinear upsampling and addition of the blended Laplacian.
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

  const F_T F_ONE = static_cast<F_T>(1.0);
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
      F_T upVal = (val00 * (F_ONE - dx) + val10 * dx) * (F_ONE - dy) + (val01 * (F_ONE - dx) + val11 * dx) * dy;
      reconImage[idxOut + c] = static_cast<T>(upVal + static_cast<F_T>(lapImage[idxOut + c]));
    }
  }
}

// =============================================================================
// Templated Host Functions: Batched Laplacian Blending
// The image size is computed as imageWidth * imageHeight * channels.
template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlend(
    const T* h_image1,
    const T* h_image2,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels, // New parameter: 3 for RGB or 4 for RGBA.
    int numLevels,
    int batchSize,
    cudaStream_t stream) {
  size_t imageSize = imageWidth * imageHeight * channels * sizeof(T);
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
  size_t sizeRGB0 = widths[0] * heights[0] * channels * batchSize * sizeof(T);
  size_t sizeMask0 = widths[0] * heights[0] * sizeof(T);
  cudaMalloc((void**)&d_gauss1[0], sizeRGB0);
  cudaMalloc((void**)&d_gauss2[0], sizeRGB0);
  cudaMalloc((void**)&d_maskPyr[0], sizeMask0);
  cudaMemcpyAsync(d_gauss1[0], h_image1, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_gauss2[0], h_image2, imageSize * batchSize, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_maskPyr[0], h_mask, maskSize, cudaMemcpyHostToDevice, stream);

  // Allocate device memory for higher pyramid levels.
  for (int level = 1; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * channels * batchSize * sizeof(T);
    size_t sizeMask = widths[level] * heights[level] * sizeof(T);
    cudaMalloc((void**)&d_gauss1[level], sizeRGB);
    cudaMalloc((void**)&d_gauss2[level], sizeRGB);
    cudaMalloc((void**)&d_maskPyr[level], sizeMask);
  }

  dim3 block(16, 16, 1);

  // 1. Build Gaussian pyramids.
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 gridRGB((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, batchSize);
    FusedBatchedDownsampleKernel<T><<<gridRGB, block, 0, stream>>>(
        d_gauss1[level],
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        batchSize,
        channels);
    {
      dim3 gridMask((widths[level + 1] + block.x - 1) / block.x, (heights[level + 1] + block.y - 1) / block.y, 1);
      BatchedDownsampleKernelMask<T><<<gridMask, block, 0, stream>>>(
          d_maskPyr[level], widths[level], heights[level], d_maskPyr[level + 1], widths[level + 1], heights[level + 1]);
    }
  }

  // 2. Build Laplacian pyramids.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * channels * batchSize * sizeof(T);
    cudaMalloc((void**)&d_lap1[level], sizeRGB);
    cudaMalloc((void**)&d_lap2[level], sizeRGB);
  }
  for (int level = 0; level < numLevels - 1; level++) {
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedComputeLaplacianKernel<T, F_T><<<grid, block, 0, stream>>>(
        d_gauss1[level],
        widths[level],
        heights[level],
        d_gauss1[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap1[level],
        batchSize,
        channels);
    BatchedComputeLaplacianKernel<T, F_T><<<grid, block, 0, stream>>>(
        d_gauss2[level],
        widths[level],
        heights[level],
        d_gauss2[level + 1],
        widths[level + 1],
        heights[level + 1],
        d_lap2[level],
        batchSize,
        channels);
  }
  int last = numLevels - 1;
  {
    size_t lastSize = widths[last] * heights[last] * channels * batchSize * sizeof(T);
    cudaMemcpyAsync(d_lap1[last], d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(d_lap2[last], d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream);
  }

  // 3. Blend the Laplacian pyramids using the shared mask.
  for (int level = 0; level < numLevels; level++) {
    size_t sizeRGB = widths[level] * heights[level] * channels * batchSize * sizeof(T);
    cudaMalloc((void**)&d_blend[level], sizeRGB);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedBlendKernel<T, F_T><<<grid, block, 0, stream>>>(
        d_lap1[level],
        d_lap2[level],
        d_maskPyr[level],
        d_blend[level],
        widths[level],
        heights[level],
        batchSize,
        channels);
  }

  // 4. Reconstruct the final blended image.
  T* d_reconstruct = nullptr;
  cudaMalloc((void**)&d_reconstruct, widths[last] * heights[last] * channels * batchSize * sizeof(T));
  cudaMemcpyAsync(
      d_reconstruct,
      d_blend[last],
      widths[last] * heights[last] * channels * batchSize * sizeof(T),
      cudaMemcpyDeviceToDevice,
      stream);
  for (int level = numLevels - 2; level >= 0; level--) {
    T* d_temp = nullptr;
    size_t highSize = widths[level] * heights[level] * channels * batchSize * sizeof(T);
    cudaMalloc((void**)&d_temp, highSize);
    dim3 grid((widths[level] + block.x - 1) / block.x, (heights[level] + block.y - 1) / block.y, batchSize);
    BatchedReconstructKernel<T, F_T><<<grid, block, 0, stream>>>(
        d_reconstruct,
        widths[level + 1],
        heights[level + 1],
        d_blend[level],
        widths[level],
        heights[level],
        d_temp,
        batchSize,
        channels);
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

// -----------------------------------------------------------------------------
// Templated version with context, now accepting the extra "channels" parameter.
template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendWithContext(
    const T* d_image1,
    const T* d_image2,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext<T>& context,
    int channels,
    cudaStream_t stream) {
  // size_t imageSize = context.imageWidth * context.imageHeight * channels * sizeof(T);

  // Initialization: set up pyramid dimensions and allocate device memory using channels.
  if (!context.initialized) {
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < context.numLevels; i++) {
      context.widths[i] = (context.widths[i - 1] + 1) / 2;
      context.heights[i] = (context.heights[i - 1] + 1) / 2;
      assert(context.widths[i] && context.heights[i]);
    }
    for (int level = 0; level < context.numLevels; level++) {
      size_t sizeRGB = context.widths[level] * context.heights[level] * channels * context.batchSize * sizeof(T);
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
    FusedBatchedDownsampleKernel<T><<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.batchSize,
        channels);
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
    BatchedComputeLaplacianKernel<T, F_T><<<grid, block, 0, stream>>>(
        context.d_gauss1[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss1[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap1[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
    BatchedComputeLaplacianKernel<T, F_T><<<grid, block, 0, stream>>>(
        context.d_gauss2[level],
        context.widths[level],
        context.heights[level],
        context.d_gauss2[level + 1],
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_lap2[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
  }
  int last = context.numLevels - 1;
  CUDA_CHECK(cudaMemcpyAsync(
      context.d_lap1[last],
      context.d_gauss1[last],
      context.widths[last] * context.heights[last] * channels * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));
  CUDA_CHECK(cudaMemcpyAsync(
      context.d_lap2[last],
      context.d_gauss2[last],
      context.widths[last] * context.heights[last] * channels * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));

  // 3. Blend the Laplacian pyramids.
  for (int level = 0; level < context.numLevels; level++) {
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedBlendKernel<T, F_T><<<grid, block, 0, stream>>>(
        context.d_lap1[level],
        context.d_lap2[level],
        context.d_maskPyr[level],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
  }

  // 4. Reconstruct the final image.
  T* d_reconstruct = nullptr;
  if (!context.initialized) {
    if (context.numLevels > 1) {
      size_t sz = context.widths[last] * context.heights[last] * channels * sizeof(T) * context.batchSize;
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
      context.widths[last] * context.heights[last] * channels * sizeof(T) * context.batchSize,
      cudaMemcpyDeviceToDevice,
      stream));
  for (int level = context.numLevels - 2; level >= 0; level--) {
    T* d_temp = nullptr;
    size_t highSize = context.widths[level] * context.heights[level] * channels * sizeof(T) * context.batchSize;
    if (!context.initialized) {
      if (level > 0) {
        CUDA_CHECK(cudaMalloc((void**)&d_temp, highSize));
        context.allocation_size += highSize;
        assert(!context.d_resonstruct[level]);
        context.d_resonstruct[level] = d_temp;
      } else {
        d_temp = d_output;
      }
    } else {
      d_temp = (level > 0) ? context.d_resonstruct[level] : d_output;
    }
    dim3 grid(
        (context.widths[level] + block.x - 1) / block.x,
        (context.heights[level] + block.y - 1) / block.y,
        context.batchSize);
    BatchedReconstructKernel<T, F_T><<<grid, block, 0, stream>>>(
        d_reconstruct,
        context.widths[level + 1],
        context.heights[level + 1],
        context.d_blend[level],
        context.widths[level],
        context.heights[level],
        d_temp,
        context.batchSize,
        channels);
    CUDA_CHECK(cudaGetLastError());
    d_reconstruct = d_temp;
  }
  assert(d_reconstruct == d_output);
  context.initialized = true;

  return cudaSuccess;
}

//------------------------------------------------------------------------------
// Explicit template instantiations for supported data types.
// The macros are updated to include the extra "channels" parameter.

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(T)         \
  template cudaError_t cudaBatchedLaplacianBlend<T, float>( \
      const T* h_image1,                                    \
      const T* h_image2,                                    \
      const T* h_mask,                                      \
      T* h_output,                                          \
      int imageWidth,                                       \
      int imageHeight,                                      \
      int channels,                                         \
      int numLevels,                                        \
      int batchSize,                                        \
      cudaStream_t stream);

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(T)       \
  template cudaError_t cudaBatchedLaplacianBlendWithContext<T, float>( \
      const T* d_image1,                                               \
      const T* d_image2,                                               \
      const T* d_mask,                                                 \
      T* d_output,                                                     \
      CudaBatchLaplacianBlendContext<T>& context,                      \
      int channels,                                                    \
      cudaStream_t stream);

INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(half)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(float)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(unsigned char)
// INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(__half)
// INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND(__nv_bfloat16)

INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(half)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(float)
INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(unsigned char)
// INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(__half)
// INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT(__nv_bfloat16)
