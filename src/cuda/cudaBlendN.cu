// cudaBlendN.cu
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include "cudaBlendN.h"

#define CUDA_CHECK(call)                                                                                    \
  do {                                                                                                      \
    cudaError_t e = (call);                                                                                 \
    if (e != cudaSuccess) {                                                                                 \
      std::cerr << "CUDA error " << cudaGetErrorString(e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
      return e;                                                                                             \
    }                                                                                                       \
  } while (0)
namespace {
// -----------------------------------------------------------------------------
// 1) Downsample ANY N_IMAGES-channel mask by simple averaging
// -----------------------------------------------------------------------------
template <typename T, int N_IMAGES>
__global__ void FusedBatchedDownsampleMaskN(
    const T* __restrict__ input, // [inH × inW × N_IMAGES]
    int inW,
    int inH,
    T* __restrict__ output, // [outH × outW × N_IMAGES]
    int outW,
    int outH) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outW || y >= outH)
    return;

  int inX = x * 2, inY = y * 2;
  float sums[N_IMAGES] = {0.0f};
  int count = 0;
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      int ix = inX + dx, iy = inY + dy;
      if (ix < inW && iy < inH) {
        int idx = (iy * inW + ix) * N_IMAGES;
        for (int c = 0; c < N_IMAGES; ++c)
          sums[c] += static_cast<float>(input[idx + c]);
        ++count;
      }
    }
  int outIdx = (y * outW + x) * N_IMAGES;
  for (int c = 0; c < N_IMAGES; ++c)
    output[outIdx + c] = static_cast<T>(sums[c] / count);
}

// -----------------------------------------------------------------------------
// 2) Fused downsample kernel for N_IMAGES inputs (RGB or RGBA-like channels)
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
__global__ void FusedBatchedDownsampleKernelN(
    const T* const* inputs,
    int inW,
    int inH,
    T* const* outputs,
    int outW,
    int outH,
    int batchSize) {
  constexpr int sumCh = (CHANNELS > 3 ? 3 : CHANNELS);
  int b = blockIdx.z;
  if (b >= batchSize)
    return;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= outW || y >= outH)
    return;

  // accumulators
  F_T sums[N_IMAGES][3] = {};
  int counts[N_IMAGES] = {};
  T alphas[N_IMAGES] = {};

  int inX = x * 2, inY = y * 2;
  for (int dy = 0; dy < 2; ++dy)
    for (int dx = 0; dx < 2; ++dx) {
      int ix = inX + dx, iy = inY + dy;
      if (ix >= inW || iy >= inH)
        continue;
      int base = (iy * inW + ix) * CHANNELS;
      for (int i = 0; i < N_IMAGES; ++i) {
        const T* p = inputs[i] + b * (inW * inH * CHANNELS) + base;
        bool keep = true;
        if constexpr (CHANNELS == 4) {
          T a = p[3];
          alphas[i] = max(alphas[i], a);
          keep = (a != 0);
        }
        if (!keep)
          continue;
        for (int c = 0; c < sumCh; ++c)
          sums[i][c] += static_cast<F_T>(p[c]);
        ++counts[i];
      }
    }

  int outBase = b * (outW * outH * CHANNELS) + (y * outW + x) * CHANNELS;
  for (int i = 0; i < N_IMAGES; ++i) {
    T* o = outputs[i] + outBase;
    if (counts[i] > 0) {
      for (int c = 0; c < sumCh; ++c)
        o[c] = static_cast<T>(sums[i][c] / counts[i]);
    } else {
      for (int c = 0; c < sumCh; ++c)
        o[c] = T(0);
    }
    if constexpr (CHANNELS == 4) {
      o[3] = alphas[i];
    }
  }
}

// -----------------------------------------------------------------------------
// 3) Compute Laplacian for N_IMAGES inputs
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
__global__ void BatchedComputeLaplacianKernelN(
    const T* const* gaussHigh,
    int highW,
    int highH,
    const T* const* gaussLow,
    int lowW,
    int lowH,
    T* const* laplacian,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highW || y >= highH)
    return;

  // This upsample is always 2x (top-left aligned), so dx/dy are either 0 or 0.5.
  const int gxi = x >> 1;
  const int gyi = y >> 1;
  const int gxi1 = min(gxi + 1, lowW - 1);
  const int gyi1 = min(gyi + 1, lowH - 1);

  const F_T wx1 = (x & 1) ? F_T(0.5) : F_T(0);
  const F_T wx0 = F_T(1) - wx1;
  const F_T wy1 = (y & 1) ? F_T(0.5) : F_T(0);
  const F_T wy0 = F_T(1) - wy1;

  const F_T w00 = wx0 * wy0;
  const F_T w10 = wx1 * wy0;
  const F_T w01 = wx0 * wy1;
  const F_T w11 = wx1 * wy1;

  const int baseH = b * (highW * highH * CHANNELS) + (y * highW + x) * CHANNELS;
  const int base00 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi) * CHANNELS;
  const int base10 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi1) * CHANNELS;
  const int base01 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi) * CHANNELS;
  const int base11 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi1) * CHANNELS;

  for (int i = 0; i < N_IMAGES; ++i) {
    const T* high = gaussHigh[i] + baseH;
    const T* low = gaussLow[i];
    T* out = laplacian[i] + baseH;

    if constexpr (CHANNELS == 4) {
      const bool v00 = static_cast<F_T>(low[base00 + 3]) != F_T(0);
      const bool v10 = static_cast<F_T>(low[base10 + 3]) != F_T(0);
      const bool v01 = static_cast<F_T>(low[base01 + 3]) != F_T(0);
      const bool v11 = static_cast<F_T>(low[base11 + 3]) != F_T(0);

      const F_T w00v = v00 ? w00 : F_T(0);
      const F_T w10v = v10 ? w10 : F_T(0);
      const F_T w01v = v01 ? w01 : F_T(0);
      const F_T w11v = v11 ? w11 : F_T(0);

      const F_T sumW = w00v + w10v + w01v + w11v;
      const F_T invW = (sumW > F_T(0)) ? (F_T(1) / sumW) : F_T(0);

#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const F_T sumV = static_cast<F_T>(low[base00 + c]) * w00v + static_cast<F_T>(low[base10 + c]) * w10v +
            static_cast<F_T>(low[base01 + c]) * w01v + static_cast<F_T>(low[base11 + c]) * w11v;
        const F_T up = sumV * invW;
        out[c] = static_cast<T>(static_cast<F_T>(high[c]) - up);
      }

      out[3] = high[3];
    } else {
#pragma unroll
      for (int c = 0; c < CHANNELS; ++c) {
        const F_T up = static_cast<F_T>(low[base00 + c]) * w00 + static_cast<F_T>(low[base10 + c]) * w10 +
            static_cast<F_T>(low[base01 + c]) * w01 + static_cast<F_T>(low[base11 + c]) * w11;
        out[c] = static_cast<T>(static_cast<F_T>(high[c]) - up);
      }
    }
  }
}

// -----------------------------------------------------------------------------
// 4) Blend N_IMAGES Laplacians with N_IMAGES-channel mask
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
__global__ void BatchedBlendKernelN(
    const T* const* laplacians,
    const T* mask, // [H×W×N_IMAGES]
    T* out, // [batch×H×W×CHANNELS]
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

  int pix = y * width + x;
  F_T w[N_IMAGES], sumW = 0;
  for (int i = 0; i < N_IMAGES; ++i) {
    w[i] = static_cast<F_T>(mask[pix * N_IMAGES + i]);
    sumW += w[i];
  }
  if (sumW > 0)
    for (int i = 0; i < N_IMAGES; ++i)
      w[i] /= sumW;

  int base = b * (width * height * CHANNELS) + pix * CHANNELS;
  if constexpr (CHANNELS == 4) {
    F_T alpha[N_IMAGES];
#pragma unroll
    for (int i = 0; i < N_IMAGES; ++i) {
      const T* L = laplacians[i] + base;
      alpha[i] = static_cast<F_T>(L[3]);
      if (alpha[i] == F_T(0)) {
        w[i] = F_T(0);
      }
    }

    F_T sumW2 = 0;
#pragma unroll
    for (int i = 0; i < N_IMAGES; ++i) {
      sumW2 += w[i];
    }
    if (sumW2 > F_T(0)) {
      const F_T inv = F_T(1) / sumW2;
#pragma unroll
      for (int i = 0; i < N_IMAGES; ++i) {
        w[i] *= inv;
      }
    } else if (sumW > F_T(0)) {
      // Mask selected only transparent pixels; fall back to the highest-alpha contributor.
      int best = -1;
      F_T best_alpha = F_T(0);
#pragma unroll
      for (int i = 0; i < N_IMAGES; ++i) {
        if (alpha[i] > best_alpha) {
          best_alpha = alpha[i];
          best = i;
        }
      }
      if (best >= 0 && best_alpha > F_T(0)) {
#pragma unroll
        for (int i = 0; i < N_IMAGES; ++i) {
          w[i] = (i == best) ? F_T(1) : F_T(0);
        }
      }
    }

#pragma unroll
    for (int c = 0; c < 3; ++c) {
      F_T acc = 0;
#pragma unroll
      for (int i = 0; i < N_IMAGES; ++i) {
        const T* L = laplacians[i] + base;
        acc += w[i] * static_cast<F_T>(L[c]);
      }
      out[base + c] = static_cast<T>(acc);
    }

    F_T accA = 0;
#pragma unroll
    for (int i = 0; i < N_IMAGES; ++i) {
      accA += w[i] * alpha[i];
    }
    out[base + 3] = static_cast<T>(accA);
  } else {
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      F_T acc = 0;
#pragma unroll
      for (int i = 0; i < N_IMAGES; ++i) {
        const T* L = laplacians[i] + base;
        acc += w[i] * static_cast<F_T>(L[c]);
      }
      out[base + c] = static_cast<T>(acc);
    }
  }
}

// -----------------------------------------------------------------------------
// 5) Reconstruction: upsample low + add Laplacian for the blended pyramid
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int CHANNELS>
__global__ void BatchedReconstructKernelBlended(
    const T* __restrict__ lowerRes,
    int lowW,
    int lowH,
    const T* __restrict__ lap,
    int highW,
    int highH,
    T* __restrict__ out,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highW || y >= highH)
    return;

  // This upsample is always 2x (top-left aligned), so dx/dy are either 0 or 0.5.
  const int gxi = x >> 1;
  const int gyi = y >> 1;
  const int gxi1 = min(gxi + 1, lowW - 1);
  const int gyi1 = min(gyi + 1, lowH - 1);

  // Output pixel location
  const int baseH = b * (highW * highH * CHANNELS) + (y * highW + x) * CHANNELS;

  // Lower-res 2x2 base offsets
  const int base00 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi) * CHANNELS;
  const int base10 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi1) * CHANNELS;
  const int base01 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi) * CHANNELS;
  const int base11 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi1) * CHANNELS;

  const F_T wx1 = (x & 1) ? F_T(0.5) : F_T(0);
  const F_T wx0 = F_T(1) - wx1;
  const F_T wy1 = (y & 1) ? F_T(0.5) : F_T(0);
  const F_T wy0 = F_T(1) - wy1;

  const F_T w00 = wx0 * wy0;
  const F_T w10 = wx1 * wy0;
  const F_T w01 = wx0 * wy1;
  const F_T w11 = wx1 * wy1;

  if constexpr (CHANNELS == 4) {
    const bool v00 = static_cast<F_T>(lowerRes[base00 + 3]) != F_T(0);
    const bool v10 = static_cast<F_T>(lowerRes[base10 + 3]) != F_T(0);
    const bool v01 = static_cast<F_T>(lowerRes[base01 + 3]) != F_T(0);
    const bool v11 = static_cast<F_T>(lowerRes[base11 + 3]) != F_T(0);

    const F_T w00v = v00 ? w00 : F_T(0);
    const F_T w10v = v10 ? w10 : F_T(0);
    const F_T w01v = v01 ? w01 : F_T(0);
    const F_T w11v = v11 ? w11 : F_T(0);

    const F_T sumW = w00v + w10v + w01v + w11v;
    const F_T invW = (sumW > F_T(0)) ? (F_T(1) / sumW) : F_T(0);

#pragma unroll
    for (int c = 0; c < 3; ++c) {
      const F_T sumV = static_cast<F_T>(lowerRes[base00 + c]) * w00v + static_cast<F_T>(lowerRes[base10 + c]) * w10v +
          static_cast<F_T>(lowerRes[base01 + c]) * w01v + static_cast<F_T>(lowerRes[base11 + c]) * w11v;
      const F_T upVal = sumV * invW;
      out[baseH + c] = static_cast<T>(upVal + static_cast<F_T>(lap[baseH + c]));
    }
    out[baseH + 3] = lap[baseH + 3];
  } else {
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      const F_T upVal = static_cast<F_T>(lowerRes[base00 + c]) * w00 + static_cast<F_T>(lowerRes[base10 + c]) * w10 +
          static_cast<F_T>(lowerRes[base01 + c]) * w01 + static_cast<F_T>(lowerRes[base11 + c]) * w11;
      out[baseH + c] = static_cast<T>(upVal + static_cast<F_T>(lap[baseH + c]));
    }
  }
}

} // namespace

// -----------------------------------------------------------------------------
// 6) Host wrapper: allocate, copy, build pyramids, blend, reconstruct, free
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendN(
    const std::vector<const T*>& h_imagePtrs,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int maxLevels,
    int batchSize,
    cudaStream_t stream) {
  // 1) Validate
  if ((int)h_imagePtrs.size() != N_IMAGES)
    return cudaErrorInvalidValue;
  if (maxLevels < 1)
    return cudaErrorInvalidValue;

  // 2) Compute pyramid sizes
  std::vector<int> widths(maxLevels), heights(maxLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int lvl = 1; lvl < maxLevels; ++lvl) {
    int w = (widths[lvl - 1] + 1) / 2;
    int h = (heights[lvl - 1] + 1) / 2;
    if (w < 2 || h < 2) {
      maxLevels = lvl;
      break;
    }
    widths[lvl] = w;
    heights[lvl] = h;
  }
  int numLevels = maxLevels;
  widths.resize(numLevels);
  heights.resize(numLevels);

  // 3) Allocate device pointers arrays
  std::array<std::vector<T*>, N_IMAGES> d_gauss, d_lap;
  for (int i = 0; i < N_IMAGES; ++i) {
    d_gauss[i].resize(numLevels);
    d_lap[i].resize(numLevels);
  }
  std::vector<T*> d_maskPyr(numLevels);
  std::vector<T*> d_blend(numLevels);
  std::vector<T*> d_reconstruct(numLevels);

  // 4) Level-0 alloc + copy
  size_t level0ImgBytes = size_t(widths[0]) * heights[0] * CHANNELS * batchSize * sizeof(T);
  size_t level0MaskBytes = size_t(widths[0]) * heights[0] * N_IMAGES * sizeof(T);
  for (int i = 0; i < N_IMAGES; ++i) {
    CUDA_CHECK(cudaMalloc(&d_gauss[i][0], level0ImgBytes));
    CUDA_CHECK(cudaMemcpyAsync(d_gauss[i][0], h_imagePtrs[i], level0ImgBytes, cudaMemcpyHostToDevice, stream));
  }
  CUDA_CHECK(cudaMalloc(&d_maskPyr[0], level0MaskBytes));
  CUDA_CHECK(cudaMemcpyAsync(d_maskPyr[0], h_mask, level0MaskBytes, cudaMemcpyHostToDevice, stream));

  // 5) Allocate higher levels
  for (int lvl = 1; lvl < numLevels; ++lvl) {
    size_t imgBytes = size_t(widths[lvl]) * heights[lvl] * CHANNELS * batchSize * sizeof(T);
    size_t maskBytes = size_t(widths[lvl]) * heights[lvl] * N_IMAGES * sizeof(T);
    for (int i = 0; i < N_IMAGES; ++i)
      CUDA_CHECK(cudaMalloc(&d_gauss[i][lvl], imgBytes));
    CUDA_CHECK(cudaMalloc(&d_maskPyr[lvl], maskBytes));
  }

  dim3 blk(16, 16);
  // 6) Build Gaussian pyramids
  for (int lvl = 0; lvl < numLevels - 1; ++lvl) {
    int inW = widths[lvl], inH = heights[lvl];
    int outW = widths[lvl + 1], outH = heights[lvl + 1];
    dim3 gridImg((outW + 15) / 16, (outH + 15) / 16, batchSize);
    // prepare device-array-of-ptrs
    std::vector<const T*> h_in(N_IMAGES), h_out(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
      h_in[i] = d_gauss[i][lvl];
      h_out[i] = d_gauss[i][lvl + 1];
    }
    const T** d_in;
    T** d_out;
    CUDA_CHECK(cudaMalloc(&d_in, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMalloc(&d_out, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_out, h_out.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    FusedBatchedDownsampleKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridImg, blk, 0, stream>>>(d_in, inW, inH, d_out, outW, outH, batchSize);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_in);
    cudaFree(d_out);

    // mask downsample
    dim3 gridMask((outW + 15) / 16, (outH + 15) / 16);
    FusedBatchedDownsampleMaskN<T, N_IMAGES>
        <<<gridMask, blk, 0, stream>>>(d_maskPyr[lvl], inW, inH, d_maskPyr[lvl + 1], outW, outH);
    CUDA_CHECK(cudaGetLastError());
  }

  // 7) Allocate Laplacian & blend & reconstruct buffers
  for (int lvl = 0; lvl < numLevels; ++lvl) {
    size_t imgBytes = size_t(widths[lvl]) * heights[lvl] * CHANNELS * batchSize * sizeof(T);
    for (int i = 0; i < N_IMAGES; ++i)
      CUDA_CHECK(cudaMalloc(&d_lap[i][lvl], imgBytes));
    CUDA_CHECK(cudaMalloc(&d_blend[lvl], imgBytes));
    CUDA_CHECK(cudaMalloc(&d_reconstruct[lvl], imgBytes));
  }

  // 8) Compute Laplacians
  for (int lvl = 0; lvl < numLevels - 1; ++lvl) {
    int wH = widths[lvl], hH = heights[lvl];
    int wL = widths[lvl + 1], hL = heights[lvl + 1];
    dim3 gridLap((wH + 15) / 16, (hH + 15) / 16, batchSize);

    // prepare ptr arrays
    std::vector<const T*> h_high(N_IMAGES), h_low(N_IMAGES);
    std::vector<T*> h_lap(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
      h_high[i] = d_gauss[i][lvl];
      h_low[i] = d_gauss[i][lvl + 1];
      h_lap[i] = d_lap[i][lvl];
    }
    const T** d_high;
    const T** d_low;
    T** d_lapArr;
    CUDA_CHECK(cudaMalloc(&d_high, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMalloc(&d_low, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMalloc(&d_lapArr, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMemcpyAsync(d_high, h_high.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_low, h_low.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lapArr, h_lap.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedComputeLaplacianKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridLap, blk, 0, stream>>>(d_high, wH, hH, d_low, wL, hL, d_lapArr, batchSize);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_high);
    cudaFree(d_low);
    cudaFree(d_lapArr);
  }

  // 9) Blend pyramids
  for (int lvl = 0; lvl < numLevels; ++lvl) {
    int w = widths[lvl], h = heights[lvl];
    dim3 gridBlend((w + 15) / 16, (h + 15) / 16, batchSize);

    std::vector<const T*> h_lap(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
      h_lap[i] = (lvl + 1 == numLevels) ? d_gauss[i][lvl] : d_lap[i][lvl];
    }
    const T** d_lapArr;
    CUDA_CHECK(cudaMalloc(&d_lapArr, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMemcpyAsync(d_lapArr, h_lap.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedBlendKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridBlend, blk, 0, stream>>>(d_lapArr, d_maskPyr[lvl], d_blend[lvl], w, h, batchSize);
    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_lapArr);
  }

  // 10) Reconstruct bottom-up
  for (int lvl = numLevels - 2; lvl >= 0; --lvl) {
    int wH = widths[lvl], hH = heights[lvl];
    int wL = widths[lvl + 1], hL = heights[lvl + 1];
    dim3 gridRecon((wH + 15) / 16, (hH + 15) / 16, batchSize);
    const T* d_low = (lvl + 1 == numLevels - 1) ? d_blend[lvl + 1] : d_reconstruct[lvl + 1];
    BatchedReconstructKernelBlended<T, F_T, CHANNELS>
        <<<gridRecon, blk, 0, stream>>>(d_low, wL, hL, d_blend[lvl], wH, hH, d_reconstruct[lvl], batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  // 11) Copy top-level to host if not already (it is h_output if lvl==0)
  size_t topBytes = size_t(imageWidth) * imageHeight * CHANNELS * batchSize * sizeof(T);
  CUDA_CHECK(cudaMemcpyAsync(
      h_output, (numLevels == 1) ? d_blend[0] : d_reconstruct[0], topBytes, cudaMemcpyDeviceToHost, stream));

  // 12) Free all device memory
  for (int lvl = 0; lvl < numLevels; ++lvl) {
    for (int i = 0; i < N_IMAGES; ++i) {
      cudaFree(d_gauss[i][lvl]);
      cudaFree(d_lap[i][lvl]);
    }
    cudaFree(d_maskPyr[lvl]);
    cudaFree(d_blend[lvl]);
    cudaFree(d_reconstruct[lvl]);
  }
  return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// 7) Host wrapper using pre-allocated context
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendWithContextN(
    const std::vector<const T*>& d_imagePtrs, // device pointers in[0..N_IMAGES)
    const T* d_mask, // [H×W×N_IMAGES]
    T* d_output, // [batch×H×W×CHANNELS]
    CudaBatchLaplacianBlendContextN<T, N_IMAGES>& context,
    cudaStream_t stream) {
  // 1) Validate
  if ((int)d_imagePtrs.size() != N_IMAGES)
    return cudaErrorInvalidValue;
  if (context.numLevels < 1)
    return cudaErrorInvalidValue;

  // 2) Precompute sizes
  int imageWidth = context.imageWidth;
  int imageHeight = context.imageHeight;
  int batchSize = context.batchSize;
  int maxLevels = context.numLevels;

  std::vector<int> widths(maxLevels), heights(maxLevels);
  widths[0] = imageWidth;
  heights[0] = imageHeight;
  for (int lvl = 1; lvl < maxLevels; ++lvl) {
    widths[lvl] = (widths[lvl - 1] + 1) / 2;
    heights[lvl] = (heights[lvl - 1] + 1) / 2;
  }

  // Level-0 (full-res) pointers are owned externally by the caller.
  for (int i = 0; i < N_IMAGES; ++i) {
    context.d_gauss[i][0] = const_cast<T*>(d_imagePtrs[i]);
  }
  context.d_maskPyr[0] = const_cast<T*>(d_mask);

  if (!context.initialized) {
    // Allocate Gaussian + mask pyramids for levels > 0.
    for (int lvl = 1; lvl < maxLevels; ++lvl) {
      const size_t imgBytes = size_t(widths[lvl]) * heights[lvl] * CHANNELS * batchSize * sizeof(T);
      const size_t maskBytes = size_t(widths[lvl]) * heights[lvl] * N_IMAGES * sizeof(T);
      for (int i = 0; i < N_IMAGES; ++i) {
        CUDA_CHECK(cudaMalloc(&context.d_gauss[i][lvl], imgBytes));
      }
      CUDA_CHECK(cudaMalloc(&context.d_maskPyr[lvl], maskBytes));
    }

    // Allocate laplacians (all but coarsest), blend (all levels), and reconstruct (all but coarsest, lvl>0).
    for (int lvl = 0; lvl < maxLevels; ++lvl) {
      const size_t imgBytes = size_t(widths[lvl]) * heights[lvl] * CHANNELS * batchSize * sizeof(T);
      CUDA_CHECK(cudaMalloc(&context.d_blend[lvl], imgBytes));
      if (lvl < maxLevels - 1) {
        for (int i = 0; i < N_IMAGES; ++i) {
          CUDA_CHECK(cudaMalloc(&context.d_lap[i][lvl], imgBytes));
        }
        if (lvl > 0) {
          CUDA_CHECK(cudaMalloc(&context.d_reconstruct[lvl], imgBytes));
        }
      }
    }

    context.initialized = true;
  }

  const bool rebuild_mask_pyramid = (context.mask_ptr != d_mask);
  if (rebuild_mask_pyramid) {
    context.mask_ptr = d_mask;
  }

  dim3 block(16, 16);

  // 3) Build Gaussian pyramids
  for (int lvl = 0; lvl < maxLevels - 1; ++lvl) {
    int inW = widths[lvl], inH = heights[lvl];
    int outW = widths[lvl + 1], outH = heights[lvl + 1];
    dim3 gridImg((outW + 15) / 16, (outH + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = context.d_gauss[i][lvl];
      context.h_ptrsB[i] = context.d_gauss[i][lvl + 1];
    }

    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsB, context.h_ptrsB.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    // call downsample
    FusedBatchedDownsampleKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridImg, block, 0, stream>>>(context.d_ptrsA, inW, inH, context.d_ptrsB, outW, outH, batchSize);
    CUDA_CHECK(cudaGetLastError());

    if (rebuild_mask_pyramid) {
      dim3 gridMask((outW + 15) / 16, (outH + 15) / 16);
      FusedBatchedDownsampleMaskN<T, N_IMAGES>
          <<<gridMask, block, 0, stream>>>(context.d_maskPyr[lvl], inW, inH, context.d_maskPyr[lvl + 1], outW, outH);
      CUDA_CHECK(cudaGetLastError());
    }
  }

  // 4) Compute Laplacian pyramids
  for (int lvl = 0; lvl < maxLevels - 1; ++lvl) {
    int wH = widths[lvl], hH = heights[lvl];
    int wL = widths[lvl + 1], hL = heights[lvl + 1];
    dim3 gridLap((wH + 15) / 16, (hH + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = context.d_gauss[i][lvl];
      context.h_ptrsB[i] = context.d_gauss[i][lvl + 1];
      context.h_ptrsC[i] = context.d_lap[i][lvl];
    }
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsB, context.h_ptrsB.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsC, context.h_ptrsC.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedComputeLaplacianKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridLap, block, 0, stream>>>(context.d_ptrsA, wH, hH, context.d_ptrsB, wL, hL, context.d_ptrsC, batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  // 5) Blend pyramids
  for (int lvl = 0; lvl < maxLevels; ++lvl) {
    int w = widths[lvl], h = heights[lvl];
    dim3 gridBlend((w + 15) / 16, (h + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = (lvl + 1 == maxLevels) ? context.d_gauss[i][lvl] : context.d_lap[i][lvl];
    }
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedBlendKernelN<T, F_T, N_IMAGES, CHANNELS><<<gridBlend, block, 0, stream>>>(
        context.d_ptrsA, context.d_maskPyr[lvl], context.d_blend[lvl], w, h, batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  if (maxLevels == 1) {
    const size_t imgBytes = size_t(widths[0]) * heights[0] * CHANNELS * batchSize * sizeof(T);
    CUDA_CHECK(cudaMemcpyAsync(d_output, context.d_blend[0], imgBytes, cudaMemcpyDeviceToDevice, stream));
    return cudaSuccess;
  }

  // 6) Reconstruct bottom-up
  for (int lvl = maxLevels - 2; lvl >= 0; --lvl) {
    int wH = widths[lvl], hH = heights[lvl];
    int wL = widths[lvl + 1], hL = heights[lvl + 1];
    dim3 gridRecon((wH + 15) / 16, (hH + 15) / 16, batchSize);

    const T* d_low = (lvl + 1 == maxLevels - 1) ? context.d_blend[lvl + 1] : context.d_reconstruct[lvl + 1];
    T* d_out = (lvl == 0) ? d_output : context.d_reconstruct[lvl];

    BatchedReconstructKernelBlended<T, F_T, CHANNELS>
        <<<gridRecon, block, 0, stream>>>(d_low, wL, hL, context.d_blend[lvl], wH, hH, d_out, batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Explicit instantiations for float, N_IMAGES=3, CHANNELS=4 (as example)
// -----------------------------------------------------------------------------
template cudaError_t cudaBatchedLaplacianBlendN<float, float, 3, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    int,
    int,
    int,
    int,
    cudaStream_t);

// Explicit instantiations for common pixel types and N in [2..8]
// float3 (3 channels)
// Base scalar float (mask and image buffers are passed as scalar arrays)
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 2, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 2>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 3, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 3>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 4, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 4>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 5, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 5>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 6, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 6>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 7, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 7>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 8, 3>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 8>&,
    cudaStream_t);

template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 2, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 2>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 3, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 3>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 4, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 4>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 5, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 5>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 6, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 6>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 7, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 7>&,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, 8, 4>(
    const std::vector<const float*>&,
    const float*,
    float*,
    CudaBatchLaplacianBlendContextN<float, 8>&,
    cudaStream_t);
