// cudaBlend3_optimized.cu - Complete optimized implementation of 3-image Laplacian blending

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>
#include "cudaBlend3.h"
#include "cudaTypes.h"
#include "cudaUtils.cuh"

namespace cg = cooperative_groups;

// namespace hm {
// namespace pano {
// namespace cuda {

namespace {

// Constants for optimization
constexpr int TILE_SIZE = 16;
// constexpr int SHARED_MEMORY_PADDING = 1;
// constexpr int WARP_SIZE = 32;

// Macro to check CUDA calls
#define CUDA_CHECK(call)                                                                          \
  do {                                                                                            \
    cudaError_t _err = (call);                                                                    \
    if (_err != cudaSuccess) {                                                                    \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_err)); \
      return _err;                                                                                \
    }                                                                                             \
  } while (0)

// =============================================================================
// Optimized Fused Kernels
// =============================================================================

/**
 * Fused kernel that combines downsample, upsample, and Laplacian computation
 * This eliminates multiple passes over the data
 */
template <typename T, typename F_T, int CHANNELS>
__global__ void FusedPyramidConstructionKernel3(
    // Input Gaussian pyramids at current level
    const T* __restrict__ gauss1_curr,
    const T* __restrict__ gauss2_curr,
    const T* __restrict__ gauss3_curr,
    const T* __restrict__ mask_curr,
    int currWidth,
    int currHeight,
    // Output Gaussian pyramids at next level
    T* __restrict__ gauss1_next,
    T* __restrict__ gauss2_next,
    T* __restrict__ gauss3_next,
    T* __restrict__ mask_next,
    int nextWidth,
    int nextHeight,
    // Output Laplacian pyramids at current level
    T* __restrict__ lap1_curr,
    T* __restrict__ lap2_curr,
    T* __restrict__ lap3_curr,
    int batchSize) {
  // Grid-stride loop for better GPU utilization
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  const int currImageSize = currWidth * currHeight * CHANNELS;
  const int nextImageSize = nextWidth * nextHeight * CHANNELS;
  // const int currMaskSize = currWidth * currHeight * 3;
  // const int nextMaskSize = nextWidth * nextHeight * 3;

  // Shared memory for caching
  //extern __shared__ char shared_mem[];
  //F_T* s_cache = reinterpret_cast<F_T*>(shared_mem);

  // Process downsampling and Laplacian in a single pass
  for (int idx = tid; idx < currWidth * currHeight; idx += stride) {
    int y = idx / currWidth;
    int x = idx % currWidth;

    // Part 1: Downsample if this is a 2x2 block origin
    if ((x & 1) == 0 && (y & 1) == 0) {
      int nextX = x >> 1;
      int nextY = y >> 1;

      if (nextX < nextWidth && nextY < nextHeight) {
        F_T sums1[CHANNELS] = {0}, sums2[CHANNELS] = {0}, sums3[CHANNELS] = {0};
        F_T mask_sums[3] = {0};
        int count = 0;

// Average 2x2 block
#pragma unroll
        for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
          for (int dx = 0; dx < 2; ++dx) {
            int sx = x + dx;
            int sy = y + dy;
            if (sx < currWidth && sy < currHeight) {
              int srcIdx = (sy * currWidth + sx) * CHANNELS;
              int maskIdx = (sy * currWidth + sx) * 3;

#pragma unroll
              for (int c = 0; c < CHANNELS; ++c) {
                sums1[c] += static_cast<F_T>(gauss1_curr[b * currImageSize + srcIdx + c]);
                sums2[c] += static_cast<F_T>(gauss2_curr[b * currImageSize + srcIdx + c]);
                sums3[c] += static_cast<F_T>(gauss3_curr[b * currImageSize + srcIdx + c]);
              }

              mask_sums[0] += static_cast<F_T>(mask_curr[maskIdx + 0]);
              mask_sums[1] += static_cast<F_T>(mask_curr[maskIdx + 1]);
              mask_sums[2] += static_cast<F_T>(mask_curr[maskIdx + 2]);
              count++;
            }
          }
        }

        // Write downsampled values
        if (count > 0) {
          F_T inv_count = F_T(1) / F_T(count);
          int nextIdx = (nextY * nextWidth + nextX) * CHANNELS;
          int nextMaskIdx = (nextY * nextWidth + nextX) * 3;

#pragma unroll
          for (int c = 0; c < CHANNELS; ++c) {
            gauss1_next[b * nextImageSize + nextIdx + c] = static_cast<T>(sums1[c] * inv_count);
            gauss2_next[b * nextImageSize + nextIdx + c] = static_cast<T>(sums2[c] * inv_count);
            gauss3_next[b * nextImageSize + nextIdx + c] = static_cast<T>(sums3[c] * inv_count);
          }

          mask_next[nextMaskIdx + 0] = static_cast<T>(mask_sums[0] * inv_count);
          mask_next[nextMaskIdx + 1] = static_cast<T>(mask_sums[1] * inv_count);
          mask_next[nextMaskIdx + 2] = static_cast<T>(mask_sums[2] * inv_count);
        }
      }
    }

    // Synchronize to ensure downsampled values are written
    __syncthreads();

    // Part 2: Compute Laplacian using bilinear upsampling
    F_T gx = static_cast<F_T>(x) * 0.5f;
    F_T gy = static_cast<F_T>(y) * 0.5f;
    int gxi = floorf(gx);
    int gyi = floorf(gy);
    F_T dx = gx - static_cast<F_T>(gxi);
    F_T dy = gy - static_cast<F_T>(gyi);

    int gxi1 = min(gxi + 1, nextWidth - 1);
    int gyi1 = min(gyi + 1, nextHeight - 1);

    // Compute weights
    F_T w00 = (1 - dx) * (1 - dy);
    F_T w10 = dx * (1 - dy);
    F_T w01 = (1 - dx) * dy;
    F_T w11 = dx * dy;

    int currIdx = (y * currWidth + x) * CHANNELS;

// Process each channel
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      // Get current values
      F_T curr1 = static_cast<F_T>(gauss1_curr[b * currImageSize + currIdx + c]);
      F_T curr2 = static_cast<F_T>(gauss2_curr[b * currImageSize + currIdx + c]);
      F_T curr3 = static_cast<F_T>(gauss3_curr[b * currImageSize + currIdx + c]);

      // Bilinear interpolation from next level
      int idx00 = (gyi * nextWidth + gxi) * CHANNELS + c;
      int idx10 = (gyi * nextWidth + gxi1) * CHANNELS + c;
      int idx01 = (gyi1 * nextWidth + gxi) * CHANNELS + c;
      int idx11 = (gyi1 * nextWidth + gxi1) * CHANNELS + c;

      F_T up1 = w00 * static_cast<F_T>(gauss1_next[b * nextImageSize + idx00]) +
          w10 * static_cast<F_T>(gauss1_next[b * nextImageSize + idx10]) +
          w01 * static_cast<F_T>(gauss1_next[b * nextImageSize + idx01]) +
          w11 * static_cast<F_T>(gauss1_next[b * nextImageSize + idx11]);

      F_T up2 = w00 * static_cast<F_T>(gauss2_next[b * nextImageSize + idx00]) +
          w10 * static_cast<F_T>(gauss2_next[b * nextImageSize + idx10]) +
          w01 * static_cast<F_T>(gauss2_next[b * nextImageSize + idx01]) +
          w11 * static_cast<F_T>(gauss2_next[b * nextImageSize + idx11]);

      F_T up3 = w00 * static_cast<F_T>(gauss3_next[b * nextImageSize + idx00]) +
          w10 * static_cast<F_T>(gauss3_next[b * nextImageSize + idx10]) +
          w01 * static_cast<F_T>(gauss3_next[b * nextImageSize + idx01]) +
          w11 * static_cast<F_T>(gauss3_next[b * nextImageSize + idx11]);

      // Write Laplacian
      lap1_curr[b * currImageSize + currIdx + c] = static_cast<T>(curr1 - up1);
      lap2_curr[b * currImageSize + currIdx + c] = static_cast<T>(curr2 - up2);
      lap3_curr[b * currImageSize + currIdx + c] = static_cast<T>(curr3 - up3);
    }
  }
}

/**
 * Optimized blend kernel using warp-level primitives
 */
template <typename T, typename F_T, int CHANNELS>
__global__ void OptimizedBlendKernel3(
    const T* __restrict__ lap1,
    const T* __restrict__ lap2,
    const T* __restrict__ lap3,
    const T* __restrict__ mask,
    T* __restrict__ blended,
    int width,
    int height,
    int batchSize) {
  // Use cooperative groups for better synchronization
  cg::grid_group grid = cg::this_grid();

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  int b = blockIdx.z;

  if (b >= batchSize)
    return;

  const int imageSize = width * height * CHANNELS;
  // const int maskSize = width * height * 3;

  // Process multiple pixels per thread for better efficiency
  for (int pixelIdx = tid; pixelIdx < width * height; pixelIdx += stride) {
    int y = pixelIdx / width;
    int x = pixelIdx % width;

    // Load mask values
    int maskIdx = (y * width + x) * 3;
    F_T m1 = static_cast<F_T>(mask[maskIdx + 0]);
    F_T m2 = static_cast<F_T>(mask[maskIdx + 1]);
    F_T m3 = static_cast<F_T>(mask[maskIdx + 2]);

    // For RGBA, check alpha channels and normalize weights
    if constexpr (CHANNELS == 4) {
      int baseIdx = pixelIdx * CHANNELS;
      T alpha1 = lap1[b * imageSize + baseIdx + 3];
      T alpha2 = lap2[b * imageSize + baseIdx + 3];
      T alpha3 = lap3[b * imageSize + baseIdx + 3];

      if (alpha1 == T(0))
        m1 = F_T(0);
      if (alpha2 == T(0))
        m2 = F_T(0);
      if (alpha3 == T(0))
        m3 = F_T(0);

      F_T sum = m1 + m2 + m3;
      if (sum > F_T(0)) {
        F_T inv_sum = F_T(1) / sum;
        m1 *= inv_sum;
        m2 *= inv_sum;
        m3 *= inv_sum;
      }
    }

    // Blend each channel
    int baseIdx = pixelIdx * CHANNELS;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      F_T v1 = static_cast<F_T>(lap1[b * imageSize + baseIdx + c]);
      F_T v2 = static_cast<F_T>(lap2[b * imageSize + baseIdx + c]);
      F_T v3 = static_cast<F_T>(lap3[b * imageSize + baseIdx + c]);

      F_T result = m1 * v1 + m2 * v2 + m3 * v3;
      blended[b * imageSize + baseIdx + c] = static_cast<T>(result);
    }

    // Force alpha to 255 for RGBA
    if constexpr (CHANNELS == 4) {
      blended[b * imageSize + baseIdx + 3] = static_cast<T>(255);
    }
  }
}

/**
 * Optimized reconstruction kernel with shared memory caching
 */
template <typename T, typename F_T, int CHANNELS>
__global__ void OptimizedReconstructKernel3(
    const T* __restrict__ lowerRes,
    int lowWidth,
    int lowHeight,
    const T* __restrict__ lap,
    int highWidth,
    int highHeight,
    T* __restrict__ reconstruction,
    int batchSize) {
  // Shared memory for caching lower resolution data
  __shared__ F_T s_lower[TILE_SIZE + 2][TILE_SIZE + 2][CHANNELS];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = blockIdx.x * TILE_SIZE + tx;
  int y = blockIdx.y * TILE_SIZE + ty;
  int b = blockIdx.z;

  if (b >= batchSize)
    return;

  const int lowImageSize = lowWidth * lowHeight * CHANNELS;
  const int highImageSize = highWidth * highHeight * CHANNELS;

  // Load lower resolution data into shared memory with halo
  int lowX = (blockIdx.x * TILE_SIZE) / 2 - 1 + tx;
  int lowY = (blockIdx.y * TILE_SIZE) / 2 - 1 + ty;

  if (lowX >= 0 && lowX < lowWidth && lowY >= 0 && lowY < lowHeight && tx < TILE_SIZE + 2 && ty < TILE_SIZE + 2) {
    int lowIdx = (lowY * lowWidth + lowX) * CHANNELS;
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      s_lower[ty][tx][c] = static_cast<F_T>(lowerRes[b * lowImageSize + lowIdx + c]);
    }
  } else {
#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      s_lower[ty][tx][c] = F_T(0);
    }
  }

  __syncthreads();

  // Process reconstruction
  if (x < highWidth && y < highHeight) {
    // Compute position in lower resolution
    F_T gx = static_cast<F_T>(x) * 0.5f;
    F_T gy = static_cast<F_T>(y) * 0.5f;
    int gxi = floorf(gx);
    int gyi = floorf(gy);
    F_T dx = gx - static_cast<F_T>(gxi);
    F_T dy = gy - static_cast<F_T>(gyi);

    // Map to shared memory indices
    int sx0 = (gxi - (blockIdx.x * TILE_SIZE) / 2 + 1);
    int sy0 = (gyi - (blockIdx.y * TILE_SIZE) / 2 + 1);
    int sx1 = sx0 + 1;
    int sy1 = sy0 + 1;

    int highIdx = (y * highWidth + x) * CHANNELS;

#pragma unroll
    for (int c = 0; c < CHANNELS; ++c) {
      F_T upsampled = F_T(0);

      // Bilinear interpolation from shared memory
      if (sx0 >= 0 && sx1 < TILE_SIZE + 2 && sy0 >= 0 && sy1 < TILE_SIZE + 2) {
        F_T v00 = s_lower[sy0][sx0][c];
        F_T v10 = s_lower[sy0][sx1][c];
        F_T v01 = s_lower[sy1][sx0][c];
        F_T v11 = s_lower[sy1][sx1][c];

        upsampled = (v00 * (1 - dx) + v10 * dx) * (1 - dy) + (v01 * (1 - dx) + v11 * dx) * dy;
      }

      // Add Laplacian
      F_T lapVal = static_cast<F_T>(lap[b * highImageSize + highIdx + c]);
      reconstruction[b * highImageSize + highIdx + c] = static_cast<T>(upsampled + lapVal);
    }
  }
}

} // anonymous namespace

// =============================================================================
// Optimized Host Function Implementation
// =============================================================================

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendOptimized3(
    const T* d_image1,
    const T* d_image2,
    const T* d_image3,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext3<T>& context,
    int channels,
    cudaStream_t stream) {
  // Initialize context if needed
  if (!context.initialized) {
    int maxLevels = context.numLevels;

    // Validate parameters
    if (maxLevels < 1) {
      return cudaErrorInvalidValue;
    }

    // Setup pyramid dimensions
    context.widths[0] = context.imageWidth;
    context.heights[0] = context.imageHeight;

    for (int i = 1; i < maxLevels; i++) {
      int last_w = context.widths[i - 1];
      int last_h = context.heights[i - 1];

      constexpr int kSmallestAllowableSide = 2;
      if (last_w < kSmallestAllowableSide || last_h < kSmallestAllowableSide) {
        context.numLevels = i;
        break;
      }

      context.widths[i] = (last_w + 1) / 2;
      context.heights[i] = (last_h + 1) / 2;
    }

    // Allocate pyramid memory
    for (int level = 0; level < context.numLevels; level++) {
      int w = context.widths[level];
      int h = context.heights[level];
      size_t sizeImg = static_cast<size_t>(w) * h * channels * context.batchSize * sizeof(T);
      size_t sizeMask = static_cast<size_t>(w) * h * 3 * sizeof(T);

      // Allocate Laplacian and blend buffers
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap1[level], sizeImg));
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap2[level], sizeImg));
      CUDA_CHECK(cudaMalloc((void**)&context.d_lap3[level], sizeImg));
      CUDA_CHECK(cudaMalloc((void**)&context.d_blend[level], sizeImg));

      if (level > 0) {
        CUDA_CHECK(cudaMalloc((void**)&context.d_maskPyr[level], sizeMask));
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss1[level], sizeImg));
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss2[level], sizeImg));
        CUDA_CHECK(cudaMalloc((void**)&context.d_gauss3[level], sizeImg));
        CUDA_CHECK(cudaMalloc((void**)&context.d_reconstruct[level], sizeImg));
      } else {
        // Level 0 uses input pointers
        context.d_maskPyr[0] = const_cast<T*>(d_mask);
        context.d_gauss1[0] = const_cast<T*>(d_image1);
        context.d_gauss2[0] = const_cast<T*>(d_image2);
        context.d_gauss3[0] = const_cast<T*>(d_image3);
        context.d_reconstruct[0] = d_output;
      }
    }

    context.initialized = true;
  }

  // Configure kernel launch parameters
  dim3 block(256);
  const int maxBlocks = 65535;

  // Build Gaussian pyramids and compute Laplacians using fused kernel
  for (int level = 0; level < context.numLevels - 1; level++) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    int totalPixels = wH * hH;
    int numBlocks = min((totalPixels + block.x - 1) / block.x, maxBlocks);
    dim3 grid(numBlocks, 1, context.batchSize);

    size_t sharedMemSize = sizeof(F_T) * block.x * channels * 4; // Cache for multiple values

    if (channels == 3) {
      FusedPyramidConstructionKernel3<T, F_T, 3><<<grid, block, sharedMemSize, stream>>>(
          context.d_gauss1[level],
          context.d_gauss2[level],
          context.d_gauss3[level],
          context.d_maskPyr[level],
          wH,
          hH,
          context.d_gauss1[level + 1],
          context.d_gauss2[level + 1],
          context.d_gauss3[level + 1],
          context.d_maskPyr[level + 1],
          wL,
          hL,
          context.d_lap1[level],
          context.d_lap2[level],
          context.d_lap3[level],
          context.batchSize);
    } else {
      FusedPyramidConstructionKernel3<T, F_T, 4><<<grid, block, sharedMemSize, stream>>>(
          context.d_gauss1[level],
          context.d_gauss2[level],
          context.d_gauss3[level],
          context.d_maskPyr[level],
          wH,
          hH,
          context.d_gauss1[level + 1],
          context.d_gauss2[level + 1],
          context.d_gauss3[level + 1],
          context.d_maskPyr[level + 1],
          wL,
          hL,
          context.d_lap1[level],
          context.d_lap2[level],
          context.d_lap3[level],
          context.batchSize);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // Copy coarsest level Gaussian to Laplacian
  int last = context.numLevels - 1;
  size_t lastSize =
      static_cast<size_t>(context.widths[last]) * context.heights[last] * channels * context.batchSize * sizeof(T);
  CUDA_CHECK(cudaMemcpyAsync(context.d_lap1[last], context.d_gauss1[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(context.d_lap2[last], context.d_gauss2[last], lastSize, cudaMemcpyDeviceToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(context.d_lap3[last], context.d_gauss3[last], lastSize, cudaMemcpyDeviceToDevice, stream));

  // Blend Laplacian pyramids
  for (int level = 0; level < context.numLevels; level++) {
    int w = context.widths[level];
    int h = context.heights[level];

    int totalPixels = w * h;
    int numBlocks = min((totalPixels + block.x - 1) / block.x, maxBlocks);
    dim3 gridBlend(numBlocks, 1, context.batchSize);

    if (channels == 3) {
      OptimizedBlendKernel3<T, F_T, 3><<<gridBlend, block, 0, stream>>>(
          context.d_lap1[level],
          context.d_lap2[level],
          context.d_lap3[level],
          context.d_maskPyr[level],
          context.d_blend[level],
          w,
          h,
          context.batchSize);
    } else {
      OptimizedBlendKernel3<T, F_T, 4><<<gridBlend, block, 0, stream>>>(
          context.d_lap1[level],
          context.d_lap2[level],
          context.d_lap3[level],
          context.d_maskPyr[level],
          context.d_blend[level],
          w,
          h,
          context.batchSize);
    }
    CUDA_CHECK(cudaGetLastError());
  }

  // Reconstruct final image
  T* d_reconstruct = context.d_reconstruct[last];

  // Copy blended coarsest level
  CUDA_CHECK(cudaMemcpyAsync(d_reconstruct, context.d_blend[last], lastSize, cudaMemcpyDeviceToDevice, stream));

  // Reconstruct from coarse to fine
  dim3 blockRecon(TILE_SIZE, TILE_SIZE);

  for (int level = context.numLevels - 2; level >= 0; level--) {
    int wH = context.widths[level];
    int hH = context.heights[level];
    int wL = context.widths[level + 1];
    int hL = context.heights[level + 1];

    T* d_temp = (level > 0) ? context.d_reconstruct[level] : d_output;

    dim3 gridRecon((wH + blockRecon.x - 1) / blockRecon.x, (hH + blockRecon.y - 1) / blockRecon.y, context.batchSize);

    if (channels == 3) {
      OptimizedReconstructKernel3<T, F_T, 3><<<gridRecon, blockRecon, 0, stream>>>(
          d_reconstruct, wL, hL, context.d_blend[level], wH, hH, d_temp, context.batchSize);
    } else {
      OptimizedReconstructKernel3<T, F_T, 4><<<gridRecon, blockRecon, 0, stream>>>(
          d_reconstruct, wL, hL, context.d_blend[level], wH, hH, d_temp, context.batchSize);
    }
    CUDA_CHECK(cudaGetLastError());

    d_reconstruct = d_temp;
  }

  return cudaSuccess;
}

// Explicit template instantiations
template cudaError_t cudaBatchedLaplacianBlendOptimized3<float, float>(
    const float* d_image1,
    const float* d_image2,
    const float* d_image3,
    const float* d_mask,
    float* d_output,
    CudaBatchLaplacianBlendContext3<float>& context,
    int channels,
    cudaStream_t stream);

template cudaError_t cudaBatchedLaplacianBlendOptimized3<unsigned char, float>(
    const unsigned char* d_image1,
    const unsigned char* d_image2,
    const unsigned char* d_image3,
    const unsigned char* d_mask,
    unsigned char* d_output,
    CudaBatchLaplacianBlendContext3<unsigned char>& context,
    int channels,
    cudaStream_t stream);

// } // namespace cuda
// } // namespace pano
// } // namespace hm
