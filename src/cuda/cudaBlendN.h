// cudaBlendN.h
#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <vector>

/**
 * @brief Context for batched Laplacian blending of N_IMAGES inputs.
 *
 * @tparam T        Pixel type (float, uchar, etc.)
 * @tparam N_IMAGES Number of images to blend (and mask channels)
 */
template <typename T, int N_IMAGES>
struct CudaBatchLaplacianBlendContext {
  static_assert(N_IMAGES >= 1 && N_IMAGES <= 256, "N_IMAGES must be in [1..256]");

  int numLevels;
  int imageWidth, imageHeight;
  int batchSize;
  size_t allocation_size{0};

  // Gaussian pyramids for each input image
  std::array<std::vector<T*>, N_IMAGES> d_gauss;
  // Mask pyramid: each mask level is H×W×N_IMAGES
  std::vector<T*> d_maskPyr;
  // Laplacian pyramids for each input image
  std::array<std::vector<T*>, N_IMAGES> d_lap;
  // Single blended pyramid
  std::vector<T*> d_blend;
  // Reconstruction buffers
  std::vector<T*> d_reconstruct;

  bool initialized{false};

  CudaBatchLaplacianBlendContext(int w, int h, int levels, int batch)
      : numLevels(levels), imageWidth(w), imageHeight(h), batchSize(batch) {
    for (int i = 0; i < N_IMAGES; ++i) {
      d_gauss[i].assign(levels, nullptr);
      d_lap[i].assign(levels, nullptr);
    }
    d_maskPyr.assign(levels, nullptr);
    d_blend.assign(levels, nullptr);
    d_reconstruct.assign(levels, nullptr);
  }

  ~CudaBatchLaplacianBlendContext() {
    for (int lvl = 0; lvl < numLevels; ++lvl) {
      // free laplacian & blend & reconstruct
      for (int i = 0; i < N_IMAGES; ++i) {
        cudaFree(d_lap[i][lvl]);
        if (lvl > 0)
          cudaFree(d_gauss[i][lvl]);
      }
      cudaFree(d_blend[lvl]);
      cudaFree(d_reconstruct[lvl]);
      if (lvl > 0)
        cudaFree(d_maskPyr[lvl]);
    }
  }
};

// -----------------------------------------------------------------------------
// Kernel declarations
// -----------------------------------------------------------------------------
#if 0
template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
__global__ void FusedBatchedDownsampleKernelN(
    const T* const* inputs,
    int inW,
    int inH,
    T* const* outputs,
    int outW,
    int outH,
    int batchSize);

template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
__global__ void BatchedComputeLaplacianKernelN(
    const T* const* highImgs,
    int highW,
    int highH,
    const T* const* lowImgs,
    int lowW,
    int lowH,
    T* const* lapImgs,
    int batchSize);

template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
__global__ void BatchedBlendKernelN(
    const T* const* lapImgs,
    const T* mask, // H×W×N_IMAGES
    T* out, // batch×H×W×CHANNELS
    int width,
    int height,
    int batchSize);

template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
__global__ void BatchedReconstructKernelN(
    const T* const* lowImgs,
    int lowW,
    int lowH,
    T* const* lapImgs,
    int highW,
    int highH,
    T* out,
    int batchSize);
#endif
// -----------------------------------------------------------------------------
// Host API declarations
// -----------------------------------------------------------------------------

/**
 * @brief Batched Laplacian blend N_IMAGES inputs (alloc+free internally).
 *
 * @param h_imagePtrs Host pointers to each input batch image array.
 *                    Each pointer refers to batch×H×W×CHANNELS elements.
 * @param h_mask      Host pointer to mask array H×W×N_IMAGES.
 * @param h_output    Host pointer to output array batch×H×W×CHANNELS.
 * @param imageWidth  Width of full-res images.
 * @param imageHeight Height of full-res images.
 * @param maxLevels   Number of pyramid levels.
 * @param batchSize   Number of images per batch.
 * @param stream      CUDA stream to use.
 */
template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendN(
    const std::vector<const T*>& h_imagePtrs,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int maxLevels,
    int batchSize,
    cudaStream_t stream = 0);

/**
 * @brief Batched Laplacian blend using a pre-allocated context.
 *
 * @param d_imagePtrs Device pointers to each input batch image array.
 * @param d_mask      Device pointer to mask array H×W×N_IMAGES.
 * @param d_output    Device pointer for output array batch×H×W×CHANNELS.
 * @param context     Pre-initialized CudaBatchLaplacianBlendContext.
 * @param stream      CUDA stream to use.
 */
template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendWithContextN(
    const std::vector<const T*>& d_imagePtrs,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext<T, N_IMAGES>& context,
    cudaStream_t stream = 0);
