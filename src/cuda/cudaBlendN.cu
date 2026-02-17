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

  F_T gx = static_cast<F_T>(x) / 2, gy = static_cast<F_T>(y) / 2;
  int gxi = floorf(gx), gyi = floorf(gy);
  int gxi1 = min(gxi + 1, lowW - 1), gyi1 = min(gyi + 1, lowH - 1);
  F_T dx = gx - gxi, dy = gy - gyi;

  int baseH = b * (highW * highH * CHANNELS) + (y * highW + x) * CHANNELS;
  int base00 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi) * CHANNELS;
  int base10 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi1) * CHANNELS;
  int base01 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi) * CHANNELS;
  int base11 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi1) * CHANNELS;

  for (int i = 0; i < N_IMAGES; ++i) {
    const T* H = gaussHigh[i] + baseH;
    const T* L00 = gaussLow[i] + base00;
    const T* L10 = gaussLow[i] + base10;
    const T* L01 = gaussLow[i] + base01;
    const T* L11 = gaussLow[i] + base11;
    T* out = laplacian[i] + baseH;

    for (int c = 0; c < CHANNELS; ++c) {
      if (CHANNELS == 4 && c == 3) {
        out[3] = H[3];
      } else {
        F_T v00 = static_cast<F_T>(L00[c]);
        F_T v10 = static_cast<F_T>(L10[c]);
        F_T v01 = static_cast<F_T>(L01[c]);
        F_T v11 = static_cast<F_T>(L11[c]);
        F_T up;
        if constexpr (CHANNELS == 4) {
          F_T w00 = (1 - dx) * (1 - dy), w10 = dx * (1 - dy);
          F_T w01 = (1 - dx) * dy, w11 = dx * dy;
          F_T sW = 0, sV = 0;
          if (L00[3] != T(0)) {
            sW += w00;
            sV += v00 * w00;
          }
          if (L10[3] != T(0)) {
            sW += w10;
            sV += v10 * w10;
          }
          if (L01[3] != T(0)) {
            sW += w01;
            sV += v01 * w01;
          }
          if (L11[3] != T(0)) {
            sW += w11;
            sV += v11 * w11;
          }
          up = (sW > 0 ? sV / sW : F_T(0));
        } else {
          F_T v0 = v00 * (1 - dx) + v10 * dx;
          F_T v1 = v01 * (1 - dx) + v11 * dx;
          up = v0 * (1 - dy) + v1 * dy;
        }
        out[c] = static_cast<T>(static_cast<F_T>(H[c]) - up);
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

  // Alpha-aware weight renormalization: if an input pixel has alpha==0, ignore it.
  // This prevents darkening from blending in zero-valued pixels outside a remapped region.
  if constexpr (CHANNELS == 4) {
    F_T sumW2 = 0;
    for (int i = 0; i < N_IMAGES; ++i) {
      const T* L = laplacians[i] + base;
      if (static_cast<F_T>(L[3]) == F_T(0)) {
        w[i] = F_T(0);
      }
      sumW2 += w[i];
    }
    if (sumW2 > F_T(0)) {
      for (int i = 0; i < N_IMAGES; ++i) {
        w[i] /= sumW2;
      }
    } else {
      out[base + 0] = T(0);
      out[base + 1] = T(0);
      out[base + 2] = T(0);
      out[base + 3] = T(0);
      return;
    }
  }

  for (int c = 0; c < CHANNELS; ++c) {
    if (CHANNELS == 4 && c == 3) {
      // Alpha: max of contributing alphas (typically 255 if any input is valid).
      T alpha_out = T(0);
      for (int i = 0; i < N_IMAGES; ++i) {
        if (w[i] == F_T(0))
          continue;
        const T* L = laplacians[i] + base;
        alpha_out = max(alpha_out, L[3]);
      }
      out[base + 3] = alpha_out;
    } else {
      F_T acc = 0;
      for (int i = 0; i < N_IMAGES; ++i) {
        const T* L = laplacians[i] + base;
        acc += w[i] * static_cast<F_T>(L[c]);
      }
      out[base + c] = static_cast<T>(acc);
    }
  }
}

// -----------------------------------------------------------------------------
// 5) Reconstruction: upsample low + add Laplacian for N_IMAGES inputs
// -----------------------------------------------------------------------------
template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
__global__ void BatchedReconstructKernelN(
    const T* const* lowerRes,
    int lowW,
    int lowH,
    T* const* laplacians,
    int highW,
    int highH,
    T* out,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= highW || y >= highH)
    return;

  // Map to lower-res coordinates (no center alignment)
  F_T gx = static_cast<F_T>(x) / 2;
  F_T gy = static_cast<F_T>(y) / 2;
  int gxi = max(0, min(static_cast<int>(floorf(gx)), lowW - 1));
  int gyi = max(0, min(static_cast<int>(floorf(gy)), lowH - 1));
  int gxi1 = min(gxi + 1, lowW - 1);
  int gyi1 = min(gyi + 1, lowH - 1);
  F_T dx = gx - static_cast<F_T>(gxi);
  F_T dy = gy - static_cast<F_T>(gyi);

  // Output pixel location
  int baseH = b * (highW * highH * CHANNELS) + (y * highW + x) * CHANNELS;

  // Lower-res 2x2 base offsets
  int base00 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi) * CHANNELS;
  int base10 = b * (lowW * lowH * CHANNELS) + (gyi * lowW + gxi1) * CHANNELS;
  int base01 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi) * CHANNELS;
  int base11 = b * (lowW * lowH * CHANNELS) + (gyi1 * lowW + gxi1) * CHANNELS;

  // Loop over N_IMAGES
  for (int i = 0; i < N_IMAGES; ++i) {
    const T* L00 = lowerRes[i] + base00;
    const T* L10 = lowerRes[i] + base10;
    const T* L01 = lowerRes[i] + base01;
    const T* L11 = lowerRes[i] + base11;
    const T* lap = laplacians[i] + baseH;
    T* outPtr = out + baseH;

    for (int c = 0; c < CHANNELS; ++c) {
      if (CHANNELS == 4 && c == 3) {
        // Copy alpha directly
        outPtr[3] = lap[3];
        continue;
      }

      F_T sum = 0;
      F_T weightSum = 0;

      auto try_add = [&](const T* px, F_T wx, F_T wy) {
        F_T w = wx * wy;
        if (CHANNELS == 4 && static_cast<F_T>(px[3]) == F_T(0))
          return;
        sum += static_cast<F_T>(px[c]) * w;
        weightSum += w;
      };

      try_add(L00, F_T(1) - dx, F_T(1) - dy);
      try_add(L10, dx, F_T(1) - dy);
      try_add(L01, F_T(1) - dx, dy);
      try_add(L11, dx, dy);

      F_T interp = (weightSum > F_T(0)) ? (sum / weightSum) : F_T(0);
      outPtr[c] = static_cast<T>(interp + static_cast<F_T>(lap[c]));
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
    for (int i = 0; i < N_IMAGES; ++i)
      h_lap[i] = d_lap[i][lvl];
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

    std::vector<const T*> h_low(N_IMAGES), h_lap(N_IMAGES);
    for (int i = 0; i < N_IMAGES; ++i) {
      h_low[i] = (lvl + 1 == numLevels - 1 ? d_blend[lvl + 1] : d_reconstruct[lvl + 1]);
      h_lap[i] = d_blend[lvl];
    }
    const T** d_lowArr;
    T** d_lapArr;
    CUDA_CHECK(cudaMalloc(&d_lowArr, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMalloc(&d_lapArr, N_IMAGES * sizeof(T*)));
    CUDA_CHECK(cudaMemcpyAsync(d_lowArr, h_low.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_lapArr, h_lap.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedReconstructKernelN<T, F_T, N_IMAGES, CHANNELS><<<gridRecon, blk, 0, stream>>>(
        d_lowArr, wL, hL, d_lapArr, wH, hH, (lvl == 0 ? h_output : d_reconstruct[lvl]), batchSize);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_lowArr);
    cudaFree(d_lapArr);
  }

  // 11) Copy top-level to host if not already (it is h_output if lvl==0)
  size_t topBytes = size_t(imageWidth) * imageHeight * CHANNELS * batchSize * sizeof(T);
  CUDA_CHECK(cudaMemcpyAsync(h_output, h_output /* device ptr */, topBytes, cudaMemcpyDeviceToHost, stream));

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

  // Bind level-0 pointers (safe even when already initialized).
  context.d_maskPyr[0] = const_cast<T*>(d_mask);
  context.d_reconstruct[0] = d_output;
  for (int i = 0; i < N_IMAGES; ++i) {
    context.d_gauss[i][0] = const_cast<T*>(d_imagePtrs[i]);
  }

  // 2) Initialization: allocate buffers if needed
  if (!context.initialized) {
    int maxLevels = context.numLevels;
    if (maxLevels < 1)
      return cudaErrorInvalidValue;

    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;
    for (int lvl = 1; lvl < maxLevels; ++lvl) {
      int last_w = context.widths[lvl - 1];
      int last_h = context.heights[lvl - 1];
      constexpr int kSmallestAllowableSide = 2;
      if (last_w < kSmallestAllowableSide || last_h < kSmallestAllowableSide) {
        std::cerr << "Adjusting max levels from " << maxLevels << " to " << lvl << '\n' << std::flush;
        maxLevels = lvl;
        break;
      }
      context.widths[lvl] = (last_w + 1) / 2;
      context.heights[lvl] = (last_h + 1) / 2;
    }
    context.numLevels = maxLevels;
    context.widths.resize(context.numLevels);
    context.heights.resize(context.numLevels);

    for (int lvl = 0; lvl < context.numLevels; ++lvl) {
      int w = context.widths[lvl];
      int h = context.heights[lvl];
      const size_t sizeImg = static_cast<size_t>(w) * h * CHANNELS * context.batchSize * sizeof(T);
      const size_t sizeMask = static_cast<size_t>(w) * h * N_IMAGES * sizeof(T);

      for (int i = 0; i < N_IMAGES; ++i) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.d_lap[i][lvl]), sizeImg));
        context.allocation_size += sizeImg;
      }
      CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.d_blend[lvl]), sizeImg));
      context.allocation_size += sizeImg;

      if (lvl > 0) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.d_maskPyr[lvl]), sizeMask));
        context.allocation_size += sizeMask;

        for (int i = 0; i < N_IMAGES; ++i) {
          CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.d_gauss[i][lvl]), sizeImg));
          context.allocation_size += sizeImg;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context.d_reconstruct[lvl]), sizeImg));
        context.allocation_size += sizeImg;
      }
    }
  }

  const int batchSize = context.batchSize;
  dim3 block(16, 16);

  // 3) Build Gaussian pyramids
  for (int lvl = 0; lvl < context.numLevels - 1; ++lvl) {
    int inW = context.widths[lvl], inH = context.heights[lvl];
    int outW = context.widths[lvl + 1], outH = context.heights[lvl + 1];
    dim3 gridImg((outW + 15) / 16, (outH + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = context.d_gauss[i][lvl];
      context.h_ptrsC[i] = context.d_gauss[i][lvl + 1];
    }
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsC, context.h_ptrsC.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    // call downsample
    FusedBatchedDownsampleKernelN<T, F_T, N_IMAGES, CHANNELS>
        <<<gridImg, block, 0, stream>>>(context.d_ptrsA, inW, inH, context.d_ptrsC, outW, outH, batchSize);
    CUDA_CHECK(cudaGetLastError());

    // downsample mask
    dim3 gridMask((outW + 15) / 16, (outH + 15) / 16);
    FusedBatchedDownsampleMaskN<T, N_IMAGES>
        <<<gridMask, block, 0, stream>>>(context.d_maskPyr[lvl], inW, inH, context.d_maskPyr[lvl + 1], outW, outH);
    CUDA_CHECK(cudaGetLastError());
  }

  // 4) Compute Laplacian pyramids
  for (int lvl = 0; lvl < context.numLevels - 1; ++lvl) {
    int wH = context.widths[lvl], hH = context.heights[lvl];
    int wL = context.widths[lvl + 1], hL = context.heights[lvl + 1];
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

  // Last level Laplacian is just the last Gaussian level.
  const int last = context.numLevels - 1;
  const size_t lastBytes =
      static_cast<size_t>(context.widths[last]) * context.heights[last] * CHANNELS * batchSize * sizeof(T);
  for (int i = 0; i < N_IMAGES; ++i) {
    CUDA_CHECK(
        cudaMemcpyAsync(context.d_lap[i][last], context.d_gauss[i][last], lastBytes, cudaMemcpyDeviceToDevice, stream));
  }

  // 5) Blend pyramids
  for (int lvl = 0; lvl < context.numLevels; ++lvl) {
    int w = context.widths[lvl], h = context.heights[lvl];
    dim3 gridBlend((w + 15) / 16, (h + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = context.d_lap[i][lvl];
    }
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedBlendKernelN<T, F_T, N_IMAGES, CHANNELS><<<gridBlend, block, 0, stream>>>(
        context.d_ptrsA, context.d_maskPyr[lvl], context.d_blend[lvl], w, h, batchSize);
    CUDA_CHECK(cudaGetLastError());
  }

  // 6) Reconstruct bottom-up (single blended pyramid).
  if (last == 0) {
    CUDA_CHECK(cudaMemcpyAsync(d_output, context.d_blend[0], lastBytes, cudaMemcpyDeviceToDevice, stream));
    context.initialized = true;
    return cudaSuccess;
  }

  // Seed reconstruction at the smallest level.
  CUDA_CHECK(
      cudaMemcpyAsync(context.d_reconstruct[last], context.d_blend[last], lastBytes, cudaMemcpyDeviceToDevice, stream));
  T* d_reconstruct = context.d_reconstruct[last];

  for (int lvl = last - 1; lvl >= 0; --lvl) {
    int wH = context.widths[lvl], hH = context.heights[lvl];
    int wL = context.widths[lvl + 1], hL = context.heights[lvl + 1];
    dim3 gridRecon((wH + 15) / 16, (hH + 15) / 16, batchSize);

    for (int i = 0; i < N_IMAGES; ++i) {
      context.h_ptrsA[i] = d_reconstruct;
      context.h_ptrsC[i] = context.d_blend[lvl];
    }
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsA, context.h_ptrsA.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(
        context.d_ptrsC, context.h_ptrsC.data(), N_IMAGES * sizeof(T*), cudaMemcpyHostToDevice, stream));

    BatchedReconstructKernelN<T, F_T, N_IMAGES, CHANNELS><<<gridRecon, block, 0, stream>>>(
        context.d_ptrsA,
        wL,
        hL,
        context.d_ptrsC,
        wH,
        hH,
        (lvl == 0 ? d_output : context.d_reconstruct[lvl]),
        batchSize);
    CUDA_CHECK(cudaGetLastError());

    d_reconstruct = (lvl == 0 ? d_output : context.d_reconstruct[lvl]);
  }

  context.initialized = true;
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
