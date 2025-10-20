// cudaBlendN.h
#pragma once

#include <cuda_runtime.h>
#include <array>
#include <cassert>
#include <vector>

/**
 * @brief Context for batched Laplacian blending of N_IMAGES inputs.
 *
 * Adds three pre-allocated device‐pointer arrays so that no
 * per-call malloc/free of pointer-lists is needed.
 */
template <typename T, int N_IMAGES>
struct CudaBatchLaplacianBlendContextN {
  static_assert(N_IMAGES >= 1 && N_IMAGES <= 256, "N_IMAGES must be in [1..256]");

  int numLevels;
  int imageWidth, imageHeight;
  int batchSize;
  size_t allocation_size{0};

  // --- pyramids etc (unchanged) ---
  std::array<std::vector<T*>, N_IMAGES> d_gauss;
  std::vector<T*> d_maskPyr;
  std::array<std::vector<T*>, N_IMAGES> d_lap;
  std::vector<T*> d_blend;
  std::vector<T*> d_reconstruct;

  bool initialized{false};

  // --- NEW: three device‐pointer arrays, length = N_IMAGES ---
  T** d_ptrsA{nullptr}; // downsample IN  / laplacian HIGH  / blend or recon LOW
  T** d_ptrsB{nullptr}; // downsample OUT / laplacian LOW   / blend or recon LAP
  T** d_ptrsC{nullptr}; // laplacian OUT  / (unused elsewhere)

  CudaBatchLaplacianBlendContextN(int w, int h, int levels, int batch)
      : numLevels(levels),
        imageWidth(w),
        imageHeight(h),
        batchSize(batch),
        d_gauss(),
        d_maskPyr(levels, nullptr),
        d_lap(),
        d_blend(levels, nullptr),
        d_reconstruct(levels, nullptr) {
    for (int i = 0; i < N_IMAGES; ++i) {
      d_gauss[i].assign(levels, nullptr);
      d_lap[i].assign(levels, nullptr);
    }
    // allocate the three pointer‐lists **once** (ignore errors here; will surface later on kernel use)
    (void)cudaMalloc(&d_ptrsA, N_IMAGES * sizeof(T*));
    (void)cudaMalloc(&d_ptrsB, N_IMAGES * sizeof(T*));
    (void)cudaMalloc(&d_ptrsC, N_IMAGES * sizeof(T*));
  }

  ~CudaBatchLaplacianBlendContextN() {
    // free pyramids & masks as before...
    for (int lvl = 0; lvl < numLevels; ++lvl) {
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
    // free our pointer‐lists
    cudaFree(d_ptrsA);
    cudaFree(d_ptrsB);
    cudaFree(d_ptrsC);
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
 * @param context     Pre-initialized CudaBatchLaplacianBlendContextN.
 * @param stream      CUDA stream to use.
 */
template <typename T, typename F_T = float, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendWithContextN(
    const std::vector<const T*>& d_imagePtrs,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContextN<T, N_IMAGES>& context,
    cudaStream_t stream = 0);
