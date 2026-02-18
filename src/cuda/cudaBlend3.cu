// cudaBlend3.cu
#include "pthread_clock_compat.h"

#include "cudaBlend3.h"
#include "cudaTypes.h"
#include "cudaUtils.cuh"
#include "cupano/utils/imageUtils.h"
#include "cupano/utils/showImage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cassert>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <iostream>
#include <vector>

using namespace hm::cupano::cuda;

#ifndef NDEBUG
#define PRINT_STRANGE_ALPHAS
#endif

namespace {

#define SHOWIMG(_img$)                                                                                                 \
  do {                                                                                                                 \
    context.show_image(std::string(#_img$) + " level " + std::to_string(level), context._img$, level, channels, true); \
  } while (false)

#define SHOWIMGLOCAL(_img$, _level$)                                                                               \
  do {                                                                                                             \
    context.show_image(std::string(#_img$) + " level " + std::to_string(_level$), _img$, _level$, channels, true); \
  } while (false)

#define SHOWIMGLVL(_img$, _level$)                                                                            \
  do {                                                                                                        \
    context.show_image(                                                                                       \
        std::string(#_img$) + " level " + std::to_string(_level$), context._img$, (_level$), channels, true); \
  } while (false)

#define SHOWIMGLVL_SCALED(_img$, _level$, _scale$)                                                                     \
  do {                                                                                                                 \
    context.show_image(                                                                                                \
        std::string(#_img$) + " level " + std::to_string(_level$), context._img$, (_level$), channels, true, _scale$); \
  } while (false)

#define SHOWIMGLVL_SCALEDSQ(_img$, _level$, _scale$)               \
  do {                                                             \
    context.show_image(                                            \
        std::string(#_img$) + " level " + std::to_string(_level$), \
        context._img$,                                             \
        (_level$),                                                 \
        channels,                                                  \
        /*wait=*/true,                                             \
        (_scale$),                                                 \
        /*squished=*/true);                                        \
  } while (false)

template <typename T>
void print_min_max(CudaBatchLaplacianBlendContext3<T>& context, const std::vector<T*>& vec, int level, int channels) {
  std::vector<std::pair<double, double>> min_max =
      hm::utils::getMinMaxPerChannel(context.download(vec, level, channels));
  for (int i = 0; i < min_max.size(); ++i) {
    const auto& itm = min_max[i];
    std::cout << i << "] min=" << itm.first << ", max=" << itm.second << "\n";
  }
  std::cout << std::flush;
}

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

template <typename T>
inline __device__ bool is_strange_alpha(const T& alpha) {
  return alpha != T_CONST(0) && alpha != T_CONST(255);
}

// =============================================================================
// Templated CUDA Kernels for Batched Laplacian Blending of THREE images
// =============================================================================

// -----------------------------------------------------------------------------
// Fused downsample kernel for THREE images (RGB or RGBA).
// Each output pixel is computed by averaging a 2×2 block in each of the three inputs.
// "channels" is either 3 (RGB) or 4 (RGBA). Alpha channel is max-pooled.
template <typename T, typename F_T = float, int CHANNELS>
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
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outWidth || y >= outHeight)
    return;

  const int inImageSize = inWidth * inHeight * CHANNELS;
  const int outImageSize = outWidth * outHeight * CHANNELS;
  const T* in1 = input1 + b * inImageSize;
  const T* in2 = input2 + b * inImageSize;
  const T* in3 = input3 + b * inImageSize;
  T* out1 = output1 + b * outImageSize;
  T* out2 = output2 + b * outImageSize;
  T* out3 = output3 + b * outImageSize;

  int inX = x * 2;
  int inY = y * 2;

  // we only average RGB channels
  const int sumCh = CHANNELS > 3 ? 3 : CHANNELS;

  // accumulators for each image
  F_T sums1[3] = {0}, sums2[3] = {0}, sums3[3] = {0};
  int count1 = 0, count2 = 0, count3 = 0;
  // max‐pooled alpha
  T alpha1 = 0, alpha2 = 0, alpha3 = 0;

  // loop over 2×2 block
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix >= inWidth || iy >= inHeight)
        continue;

      int idx = (iy * inWidth + ix) * CHANNELS;

      // --- IMAGE 1 ---
      bool keep1 = true;
      if constexpr (CHANNELS == 4) {
        T a = in1[idx + 3];
        alpha1 = max(alpha1, a);
        keep1 = (a != 0);
      }
      if (keep1) {
        for (int c = 0; c < sumCh; ++c)
          sums1[c] += static_cast<F_T>(in1[idx + c]);
        ++count1;
      }

      // --- IMAGE 2 ---
      bool keep2 = true;
      if constexpr (CHANNELS == 4) {
        T a = in2[idx + 3];
        alpha2 = max(alpha2, a);
        keep2 = (a != 0);
      }
      if (keep2) {
        for (int c = 0; c < sumCh; ++c)
          sums2[c] += static_cast<F_T>(in2[idx + c]);
        ++count2;
      }

      // --- IMAGE 3 ---
      bool keep3 = true;
      if constexpr (CHANNELS == 4) {
        T a = in3[idx + 3];
        alpha3 = max(alpha3, a);
        keep3 = (a != 0);
      }
      if (keep3) {
        for (int c = 0; c < sumCh; ++c)
          sums3[c] += static_cast<F_T>(in3[idx + c]);
        ++count3;
      }
    }
  }

  // write out
  int outIdx = (y * outWidth + x) * CHANNELS;

  // helper lambda to finalize each image
  auto finalize = [&](T* outImg, F_T sums[3], int count, T alpha) {
    if (count > 0) {
      for (int c = 0; c < sumCh; ++c)
        outImg[outIdx + c] = static_cast<T>(sums[c] / count);
    } else {
      // no contributing pixels → zero RGB
      for (int c = 0; c < sumCh; ++c)
        outImg[outIdx + c] = T(0);
    }
    if constexpr (CHANNELS == 4) {
      outImg[outIdx + 3] = alpha; // still max‐pooled over entire block
    }
  };

  finalize(out1, sums1, count1, alpha1);
  finalize(out2, sums2, count2, alpha2);
  finalize(out3, sums3, count3, alpha3);
}

// -----------------------------------------------------------------------------
// Fused downsample kernel for a **3-channel mask**.
// Each output mask-pixel is the per-channel average over the 2×2 block.
template <typename T, int CHANNELS = 3>
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
  float sums[CHANNELS] = {
      0.0f,
      0.0f,
      0.0f,
  };
  int count = 0;

  for (int dy = 0; dy < 2; dy++) {
    for (int dx = 0; dx < 2; dx++) {
      int ix = inX + dx;
      int iy = inY + dy;
      if (ix < inWidth && iy < inHeight) {
        int idx = (iy * inWidth + ix) * CHANNELS;
#pragma unroll
        for (int c = 0; c < CHANNELS; c++) {
          sums[c] += static_cast<float>(input[idx + c]);
        }
        count++;
      }
    }
  }

  int outIdx = (y * outWidth + x) * CHANNELS;
#pragma unroll
  for (int c = 0; c < CHANNELS; c++) {
    output[outIdx + c] = static_cast<T>(sums[c] / count);
  }
}

// -----------------------------------------------------------------------------
// Batched computation of the Laplacian for **one** image (RGB or RGBA).
// We will launch this separately for each of the three Gaussian pyramids.
template <typename T, typename F_T, int CHANNELS>
__global__ void BatchedComputeLaplacianKernel(
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

  int highImageSize = highWidth * highHeight * CHANNELS;
  int lowImageSize = lowWidth * lowHeight * CHANNELS;
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

  int idxHigh = (y * highWidth + x) * CHANNELS;

#pragma unroll
  for (int c = 0; c < CHANNELS; ++c) {
    if (CHANNELS == 4 && c == 3) {
      // alpha channel: just copy high-res alpha
      lapImage[idxHigh + c] = highImage[idxHigh + c];
    } else {
      // gather neighbor indices
      int base00 = (gyi * lowWidth + gxi) * CHANNELS;
      int base10 = (gyi * lowWidth + gxi1) * CHANNELS;
      int base01 = (gyi1 * lowWidth + gxi) * CHANNELS;
      int base11 = (gyi1 * lowWidth + gxi1) * CHANNELS;
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
      if constexpr (CHANNELS == 4) {
        // compute bilinear weights
        F_T w00 = (1 - dx) * (1 - dy);
        F_T w10 = dx * (1 - dy);
        F_T w01 = (1 - dx) * dy;
        F_T w11 = dx * dy;

        // accumulate only non-transparent neighbors
        F_T sumW = 0, sumV = 0;
        // alpha offsets
        int aOff = 3;
        if (lowImage[base00 + aOff] != T(0)) {
          sumW += w00;
          sumV += v00 * w00;
        }
        if (lowImage[base10 + aOff] != T(0)) {
          sumW += w10;
          sumV += v10 * w10;
        }
        if (lowImage[base01 + aOff] != T(0)) {
          sumW += w01;
          sumV += v01 * w01;
        }
        if (lowImage[base11 + aOff] != T(0)) {
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

// -----------------------------------------------------------------------------
// Batched blend kernel for THREE images.
// For each channel, the three Laplacian pyramids are blended with a weighted average
// using a 3-channel mask (m₁, m₂, m₃). If channels==4, alpha is treated as a validity
// channel (alpha==0 => pixel invalid). Invalid pixels are excluded by zeroing their
// weights and renormalizing; if the mask selects only invalid pixels, we fall back
// to the valid contributor with the highest alpha (similar to cudaBlend.cu behavior).
template <typename T, typename F_T>
__global__ void BatchedBlendKernel3(
    const T* __restrict__ lap1,
    const T* __restrict__ lap2,
    const T* __restrict__ lap3,
    const T* __restrict__ mask, // [H×W×3], single (non‐batched) mask
    T* __restrict__ blended, // output [batch × H × W × channels]
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

  // Fetch mask weights (cast to F_T)
  F_T m1 = static_cast<F_T>(maskImage[0]);
  F_T m2 = static_cast<F_T>(maskImage[1]);
  F_T m3 = static_cast<F_T>(maskImage[2]);

  // Compute flat index into each image’s pixel (all channels)
  int idx = (y * width + x) * channels;

  // If we have 4‐channel input, check each lap‐image’s alpha‐channel (c==3).
  // If that alpha is zero, we zero out its mask‐weight. Then renormalize.
  if (channels == 4) {
    // Read each lap‐image’s alpha at this pixel
    T alpha1 = lap1Image[idx + 3];
    T alpha2 = lap2Image[idx + 3];
    T alpha3 = lap3Image[idx + 3];

    if (alpha1 == static_cast<T>(0))
      m1 = static_cast<F_T>(0);
    if (alpha2 == static_cast<T>(0))
      m2 = static_cast<F_T>(0);
    if (alpha3 == static_cast<T>(0))
      m3 = static_cast<F_T>(0);
    assert(m1 >= 0 && m1 <= 1);
    assert(m2 >= 0 && m2 <= 1);
    assert(m3 >= 0 && m3 <= 1);
    // Renormalize so that surviving weights sum to 1.0 (if any >0 remain)
    F_T sum = m1 + m2 + m3;
    if (sum > static_cast<F_T>(0)) {
      m1 /= sum;
      m2 /= sum;
      m3 /= sum;
    } else {
      // Mask selected only invalid pixels; fall back to any valid contributor.
      // This matches the 2-image behavior in cudaBlend.cu (transparent pixels lose regardless of mask).
      const T* src = nullptr;
      T amax = static_cast<T>(0);
      if (alpha1 > amax) {
        amax = alpha1;
        src = lap1Image + idx;
      }
      if (alpha2 > amax) {
        amax = alpha2;
        src = lap2Image + idx;
      }
      if (alpha3 > amax) {
        amax = alpha3;
        src = lap3Image + idx;
      }

      if (!src) {
        // All contributors invalid.
        for (int c = 0; c < channels; ++c)
          blendImage[idx + c] = static_cast<T>(0);
      } else {
        // Copy the whole pixel (RGBA) from the chosen contributor.
        for (int c = 0; c < channels; ++c)
          blendImage[idx + c] = src[c];
      }
      return;
    }
    // If sum == 0, all three were skipped; leave m1=m2=m3=0 → contribution is zero.
    // printf("d=%p, x=%d, y=%d, b=%d, m1=%f, m2=%f, m3=%f\n", blended, (int)x, (int)y, (int)b, (float)m1, (float)m2,
    // (float)m3);
    // float summation = m1 + m2 + m3;
    // if (std::abs(1.0f - summation) > 0.01f && summation > 0.01f) {
    // if (y == height * 2 / 3) {
    //   printf(
    //       "d=%p, x=%d, y=%d, b=%d, m1=%f, m2=%f, m3=%f\n",
    //       blended,
    //       (int)x,
    //       (int)y,
    //       (int)b,
    //       (float)m1,
    //       (float)m2,
    //       (float)m3);
    // }
  }

  // We only blend up to min(4,channels). In practice, for channels==4: blend_channels=4.
  const int blend_channels = min(3, channels);

  // Perform weighted sum over (up to) 4 channels; color channels (0..2) use m1/m2/m3.
  for (int c = 0; c < blend_channels; c++) {
    F_T v1 = static_cast<F_T>(lap1Image[idx + c]);
    F_T v2 = static_cast<F_T>(lap2Image[idx + c]);
    F_T v3 = static_cast<F_T>(lap3Image[idx + c]);

    // if (v1 < 0 || v1 > 255) {
    //   printf("v1=%f at w=%d, h=%d\n", (float)v1, (int)width, (int)height);
    // }

    // assert(v1 >= 0 && v1 <= 255);
    // assert(v2 >= 0 && v2 <= 255);
    // assert(v3 >= 0 && v3 <= 255);

    // v1 = clamp(F_T(0), v1, F_T(255));
    // v2 = clamp(F_T(0), v2, F_T(255));
    // v3 = clamp(F_T(0), v3, F_T(255));

    // v1 = 0;
    // v2 = 0;
    // v3 = 0;

    F_T blendedVal = m1 * v1 + m2 * v2 + m3 * v3;
    // printf("pos=%d, pix1=%f, pix2=%f, pix3=%f -> blended=%f\n", (int)idx + c, (float)v1, (float)v2, (float)v3,
    // (float)blendedVal);
    // blendedVal = clamp(F_T(0), blendedVal, F_T(255));
    // if (v < 0)
    //   printf("v=%f", (float)v);
    // else if (v > 255) v = 255;
    blendImage[idx + c] = static_cast<T>(blendedVal);
    // blendImage[idx + c] = 128;
  }

  if (channels == 4) {
    // Alpha is max pooled over contributing (non-zero weight) pixels.
    const T alpha1 = lap1Image[idx + 3];
    const T alpha2 = lap2Image[idx + 3];
    const T alpha3 = lap3Image[idx + 3];
    T alpha_out = static_cast<T>(0);
    if (m1 > static_cast<F_T>(0))
      alpha_out = max(alpha_out, alpha1);
    if (m2 > static_cast<F_T>(0))
      alpha_out = max(alpha_out, alpha2);
    if (m3 > static_cast<F_T>(0))
      alpha_out = max(alpha_out, alpha3);
    blendImage[idx + 3] = alpha_out;
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
  assert(lowerRes != reconstruction);
  assert(lap != reconstruction);

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

  // No center alignment — pure top-left pixel mapping
  F_T gx = static_cast<F_T>(x) / 2.0f;
  F_T gy = static_cast<F_T>(y) / 2.0f;

  int gxi = max(0, min(static_cast<int>(floorf(gx)), lowWidth - 1));
  int gyi = max(0, min(static_cast<int>(floorf(gy)), lowHeight - 1));
  int gxi1 = min(gxi + 1, lowWidth - 1);
  int gyi1 = min(gyi + 1, lowHeight - 1);

  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);

  int idxOut = (y * highWidth + x) * channels;

  for (int c = 0; c < channels; ++c) {
    if (channels == 4 && c == 3) {
      // Copy alpha channel from Laplacian image
      F_T alpha = static_cast<F_T>(lapImage[idxOut + c]);

#ifdef PRINT_STRANGE_ALPHAS
      if (alpha != F_T(0) && alpha != F_T(255)) {
        printf("BatchedReconstructKernel(): Strange alpha %f\n", static_cast<float>(alpha));
      }
#endif

      reconImage[idxOut + c] = static_cast<T>(alpha);
      continue;
    }

    // Gather RGBA neighbor indices (needed to check alpha)
    int idx00 = (gyi * lowWidth + gxi) * channels;
    int idx10 = (gyi * lowWidth + gxi1) * channels;
    int idx01 = (gyi1 * lowWidth + gxi) * channels;
    int idx11 = (gyi1 * lowWidth + gxi1) * channels;

    // Alpha-aware bilinear interpolation
    F_T sum = 0;
    F_T weightSum = 0;

    auto try_add = [&](int idx, F_T wx, F_T wy) {
      F_T w = wx * wy;
      if (channels == 4 && static_cast<F_T>(lowImage[idx + 3]) == F_T(0))
        return;
      sum += static_cast<F_T>(lowImage[idx + c]) * w;
      weightSum += w;
    };

    try_add(idx00, F_ONE - dx, F_ONE - dy);
    try_add(idx10, dx, F_ONE - dy);
    try_add(idx01, F_ONE - dx, dy);
    try_add(idx11, dx, dy);

    // Normalize or fall back
    F_T upVal = (weightSum > F_T(0)) ? (sum / weightSum) : F_T(0);

    // Add Laplacian detail
    F_T computedValue = upVal + static_cast<F_T>(lapImage[idxOut + c]);
    reconImage[idxOut + c] = static_cast<T>(computedValue);
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
    int maxLevels,
    int batchSize,
    cudaStream_t stream) {
  if (maxLevels < 1) {
    return cudaErrorInvalidValue;
  }

  // Compute widths/heights of each pyramid level:
  std::vector<int> widths(maxLevels), heights(maxLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int i = 1; i < maxLevels; i++) {
    int last_w = widths[i - 1];
    int last_h = heights[i - 1];
    // We can only go down so small
    constexpr int kSmallestAllowableSide = 2; // maybe useful for unit tests only
    if (last_w < kSmallestAllowableSide || last_h < kSmallestAllowableSide) {
      std::cerr << "Adjusting max levels from " << maxLevels << " to " << i << '\n' << std::flush;
      maxLevels = i;
      break;
    }
    widths[i] = (last_w + 1) / 2;
    heights[i] = (last_h + 1) / 2;
  }

  const int numLevels = maxLevels;
  widths.resize(numLevels);
  heights.resize(numLevels);

  // Size computations
  size_t imageSize = static_cast<size_t>(imageWidth) * imageHeight * channels * sizeof(T);
  // size_t maskSizeLevel = static_cast<size_t>(imageWidth) * imageHeight * 3 * sizeof(T); // 3-channel mask

  // Storage for device pointers per level:
  std::vector<T*> d_gauss1(numLevels), d_gauss2(numLevels), d_gauss3(numLevels);
  std::vector<T*> d_maskPyr(numLevels);
  std::vector<T*> d_lap1(numLevels), d_lap2(numLevels), d_lap3(numLevels);
  std::vector<T*> d_blend(numLevels);

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
    assert(channels == 3 || channels == 4);
    if (channels == 3) {
      // Downsample the three image sets
      FusedBatchedDownsampleKernel3<T, F_T, 3><<<gridImg, block, 0, stream>>>(
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
          batchSize);
    } else {
      // Downsample the three image sets
      FusedBatchedDownsampleKernel3<T, F_T, 4><<<gridImg, block, 0, stream>>>(
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
          batchSize);
    }
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
    assert(channels == 3 || channels == 4);
    if (channels == 3) {
      // Laplacian for image 1
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          d_gauss1[level], wH, hH, d_gauss1[level + 1], wL, hL, d_lap1[level], batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 2
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          d_gauss2[level], wH, hH, d_gauss2[level + 1], wL, hL, d_lap2[level], batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 3
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          d_gauss3[level], wH, hH, d_gauss3[level + 1], wL, hL, d_lap3[level], batchSize);
      CUDA_CHECK(cudaGetLastError());
    } else if (channels == 4) {
      // Laplacian for image 1
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          d_gauss1[level], wH, hH, d_gauss1[level + 1], wL, hL, d_lap1[level], batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 2
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          d_gauss2[level], wH, hH, d_gauss2[level + 1], wL, hL, d_lap2[level], batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 3
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          d_gauss3[level], wH, hH, d_gauss3[level + 1], wL, hL, d_lap3[level], batchSize);
      CUDA_CHECK(cudaGetLastError());
    } else {
      assert(false);
    }
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
    int maxLevels = context.numLevels;
    // Set up pyramid dimensions
    if (maxLevels < 1) {
      return cudaErrorInvalidValue;
    }

    // Compute widths/heights of each pyramid level:
    std::vector<int> widths(maxLevels), heights(maxLevels);
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int i = 1; i < maxLevels; i++) {
      int last_w = context.widths[i - 1];
      int last_h = context.heights[i - 1];
      // We can only go down so small
      constexpr int kSmallestAllowableSide = 2; // maybe useful for unit tests only
      if (last_w < kSmallestAllowableSide || last_h < kSmallestAllowableSide) {
        std::cerr << "Adjusting max levels from " << maxLevels << " to " << i << '\n' << std::flush;
        maxLevels = i;
        break;
      }
      context.widths[i] = (last_w + 1) / 2;
      context.heights[i] = (last_h + 1) / 2;
    }
    context.numLevels = maxLevels;
    widths.resize(context.numLevels);
    heights.resize(context.numLevels);

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

  // SHOWIMGLVL_SCALED(d_gauss1, 0, 10);
  // SHOWIMGLVL(d_gauss2, 0);
  // SHOWIMGLVL(d_gauss3, 0);

  dim3 block(16, 16, 1);

  // --------------- Build Gaussian pyramids (downsample) ---------------
  for (int level = 0; level < context.numLevels - 1; level++) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    dim3 gridImg((wL + block.x - 1) / block.x, (hL + block.y - 1) / block.y, context.batchSize);

    assert(channels == 3 || channels == 4);
    if (channels == 3) {
      // Downsample the three image sets
      // Downsample the three image sets
      FusedBatchedDownsampleKernel3<T, F_T, 3><<<gridImg, block, 0, stream>>>(
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
          context.batchSize);
    } else {
      // Downsample the three image sets
      // Downsample the three image sets
      FusedBatchedDownsampleKernel3<T, F_T, 4><<<gridImg, block, 0, stream>>>(
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
          context.batchSize);
    }

    // print_min_max(context, context.d_gauss1, 0, channels);
    // print_min_max(context, context.d_gauss1, 1, channels);

    CUDA_CHECK(cudaGetLastError());

    // SHOWIMGLVL_SCALEDSQ(d_gauss1, level, 4.0);

    // context.show_image(std::string("d_gauss1 level ") + std::to_string(level), context.d_gauss1, level, channels,
    // true); context.show_image(std::string("d_gauss2 level ") + std::to_string(level), context.d_gauss2, level,
    // channels, true); context.show_image(std::string("d_gauss3 level ") + std::to_string(level), context.d_gauss3,
    // level, channels, true);

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

    assert(channels == 3 || channels == 4);

    if (channels == 3) {
      // Laplacian for image 1
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          context.d_gauss1[level],
          wH,
          hH,
          context.d_gauss1[level + 1],
          wL,
          hL,
          context.d_lap1[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 2
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          context.d_gauss2[level],
          wH,
          hH,
          context.d_gauss2[level + 1],
          wL,
          hL,
          context.d_lap2[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 3
      BatchedComputeLaplacianKernel<T, F_T, 3><<<gridLap, block, 0, stream>>>(
          context.d_gauss3[level],
          wH,
          hH,
          context.d_gauss3[level + 1],
          wL,
          hL,
          context.d_lap3[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());
    } else if (channels == 4) {
      // Laplacian for image 1
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          context.d_gauss1[level],
          wH,
          hH,
          context.d_gauss1[level + 1],
          wL,
          hL,
          context.d_lap1[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 2
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          context.d_gauss2[level],
          wH,
          hH,
          context.d_gauss2[level + 1],
          wL,
          hL,
          context.d_lap2[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());

      // Laplacian for image 3
      BatchedComputeLaplacianKernel<T, F_T, 4><<<gridLap, block, 0, stream>>>(
          context.d_gauss3[level],
          wH,
          hH,
          context.d_gauss3[level + 1],
          wL,
          hL,
          context.d_lap3[level],
          context.batchSize);
      CUDA_CHECK(cudaGetLastError());
    } else {
      assert(false);
    }
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

/**
 * @brief Explicit template instantiations for supported data types (3-image version).
 *        Currently instantiated for float and unsigned char.
 */
#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND3(T)         \
  template cudaError_t cudaBatchedLaplacianBlend3<T, float>( \
      const T* h_image1,                                     \
      const T* h_image2,                                     \
      const T* h_image3,                                     \
      const T* h_mask,                                       \
      T* h_output,                                           \
      int imageWidth,                                        \
      int imageHeight,                                       \
      int channels,                                          \
      int maxLevels,                                         \
      int batchSize,                                         \
      cudaStream_t stream);

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT3(T)       \
  template cudaError_t cudaBatchedLaplacianBlendWithContext3<T, float>( \
      const T* d_image1,                                                \
      const T* d_image2,                                                \
      const T* d_image3,                                                \
      const T* d_mask,                                                  \
      T* d_output,                                                      \
      CudaBatchLaplacianBlendContext3<T>& context,                      \
      int channels,                                                     \
      cudaStream_t stream);

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
