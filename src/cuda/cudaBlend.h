/**
 * @file cudaBlend.h
 * @brief CUDA-accelerated batched Laplacian blending.
 *
 * This header defines the types, host functions, and templated CUDA kernels for
 * performing batched Laplacian blending on images. The blending process builds
 * Gaussian and Laplacian pyramids for two input image sets and a single shared mask,
 * blends the Laplacian pyramids, and then reconstructs the final blended images.
 *
 * The implementation supports different image data types (e.g. float, unsigned char,
 * __half, __nv_bfloat16) and uses CUDA streams for asynchronous execution.
 */

#pragma once

#include "src/pano/cudaMat.h"
#include "src/utils/showImage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

/**
 * @brief Context structure for batched Laplacian blending on images.
 *
 * This templated structure stores the parameters and device pointers required for
 * batched Laplacian blending. It maintains device memory for Gaussian and Laplacian
 * pyramid levels, as well as for intermediate blend and reconstruction results.
 *
 * Note that the level-0 device pointers (for the original full-resolution images and mask)
 * are expected to be managed externally (by the caller) and are not freed in the destructor.
 *
 * @tparam T The data type for the images (e.g. float, unsigned char, __half, __nv_bfloat16).
 */
template <typename T>
struct CudaBatchLaplacianBlendContext {
  /**
   * @brief Constructor.
   *
   * Initializes the context with the full-resolution image dimensions, the number of pyramid
   * levels, and the batch size. The vectors holding device pointers for each pyramid level
   * are pre-sized to the number of levels.
   *
   * @param image_width Width of the full-resolution image.
   * @param image_height Height of the full-resolution image.
   * @param num_levels Number of pyramid levels to build.
   * @param batch_size Number of images in the batch.
   */
  CudaBatchLaplacianBlendContext(int image_width, int image_height, int num_levels, int batch_size)
      : numLevels(num_levels),
        imageWidth(image_width),
        imageHeight(image_height),
        batchSize(batch_size),
        widths(num_levels),
        heights(num_levels),
        d_gauss1(num_levels, nullptr),
        d_gauss2(num_levels, nullptr),
        d_maskPyr(num_levels, nullptr),
        d_lap1(num_levels, nullptr),
        d_lap2(num_levels, nullptr),
        d_blend(num_levels, nullptr),
        d_resonstruct(num_levels, nullptr) {}

  /**
   * @brief Helper function to free a CUDA pointer if it is non-null.
   *
   * This function wraps cudaFree in a null-check.
   *
   * @param p Device pointer to free.
   */
  static constexpr void maybeCudaFree(void* p) {
    if (p) {
      cudaFree(p);
    }
  }

  /**
   * @brief Destructor.
   *
   * Frees all allocated device memory for the pyramid levels (except level 0, which is assumed
   * to be managed by user code). It iterates over each pyramid level and releases memory for the
   * Laplacian, blend, and reconstruction buffers. For levels greater than zero, the Gaussian and
   * mask pyramid arrays are also freed.
   */
  ~CudaBatchLaplacianBlendContext() {
    for (int level = 0; level < numLevels; level++) {
      maybeCudaFree(d_lap1[level]);
      maybeCudaFree(d_lap2[level]);
      maybeCudaFree(d_blend[level]);
      if (level) {
        // Level 0 pointers are owned by user code (passed into the function each time)
        maybeCudaFree(d_gauss1[level]);
        maybeCudaFree(d_gauss2[level]);
        maybeCudaFree(d_maskPyr[level]);
        maybeCudaFree(d_resonstruct[level]);
      }
    }
  }

  const int numLevels; ///< Number of pyramid levels.
  const int imageWidth; ///< Width of the full-resolution image.
  const int imageHeight; ///< Height of the full-resolution image.
  const int batchSize; ///< Number of images in the batch.
  size_t allocation_size{0}; ///< Total allocated device memory size (in bytes).

  std::vector<int> widths; ///< Width of images at each pyramid level.
  std::vector<int> heights; ///< Height of images at each pyramid level.
  std::vector<T*> d_gauss1; ///< Device pointers for the Gaussian pyramid (first image set).
  std::vector<T*> d_gauss2; ///< Device pointers for the Gaussian pyramid (second image set).
  std::vector<T*> d_maskPyr; ///< Device pointers for the downsampled shared mask.
  std::vector<T*> d_lap1; ///< Device pointers for the Laplacian pyramid (first image set).
  std::vector<T*> d_lap2; ///< Device pointers for the Laplacian pyramid (second image set).
  std::vector<T*> d_blend; ///< Device pointers for the blended Laplacian pyramid.
  std::vector<T*> d_resonstruct; ///< Device pointers for temporary reconstruction buffers.
  bool initialized{false}; ///< Flag indicating whether the context has been initialized.

  // Must link to utils for this
  inline void displayPyramids(int channels, float scale, bool wait) const;
  inline void displayLevel(int level, const std::vector<T*>& surfaces, int channels, float scale, bool wait) const;

  void show_image(
      const std::string& label,
      const std::vector<T*>& vec_d_ptrs,
      int level,
      int channels,
      bool wait,
      float scale = 0.0f,
      bool squish = false);
  void show_image(
      const std::string& label,
      const T* vec_d_ptrs,
      int level,
      int channels,
      bool wait,
      float scale = 0.0f,
      bool squish = false);

  cv::Mat download(const std::vector<T*>& vec_d_ptrs, int level, int channels) const;
};

template <typename T>
cv::Mat CudaBatchLaplacianBlendContext<T>::download(const std::vector<T*>& vec_d_ptrs, int level, int channels) const {
  T* d_ptr = vec_d_ptrs.at(level);
  if (channels == 3) {
    assert(sizeof(T) * channels == sizeof(float3));
    hm::CudaMat<float3> mat((float3*)d_ptr, 1, widths.at(level), heights.at(level));
    return mat.download();
  } else {
    assert(sizeof(T) * channels == sizeof(float4));
    hm::CudaMat<float4> mat((float4*)d_ptr, 1, widths.at(level), heights.at(level));
    return mat.download();
  }
}

template <typename T>
inline void CudaBatchLaplacianBlendContext<T>::show_image(
    const std::string& label,
    const T* d_ptr,
    int level,
    int channels,
    bool wait,
    float scale,
    bool squish) {
  if (channels == 3) {
    assert(sizeof(T) * channels == sizeof(float3));
    hm::CudaMat<float3> mat((float3*)d_ptr, 1, widths.at(level), heights.at(level));
    hm::utils::show_image(label, mat.download(), wait, scale, squish);
  } else {
    assert(sizeof(T) * channels == sizeof(float4));
    hm::CudaMat<float4> mat((float4*)d_ptr, 1, widths.at(level), heights.at(level));
    hm::utils::show_image(label, mat.download(), wait, scale, squish);
  }
}

template <typename T>
inline void CudaBatchLaplacianBlendContext<T>::show_image(
    const std::string& label,
    const std::vector<T*>& vec_d_ptrs,
    int level,
    int channels,
    bool wait,
    float scale,
    bool squish) {
  show_image(label, vec_d_ptrs.at(level), level, channels, wait, scale, squish);
}

/**
 * @brief Performs batched Laplacian blending on images.
 *
 * This host function copies the input full-resolution images and shared mask from host to device,
 * builds Gaussian and Laplacian pyramids for both image sets, blends the Laplacian pyramids using
 * the shared mask, reconstructs the final blended images, and copies the result back to host memory.
 *
 * @tparam T The image data type.
 * @param h_image1 Host pointer to the first set of full-resolution images (batched layout).
 * @param h_image2 Host pointer to the second set of full-resolution images (batched layout).
 * @param h_mask Host pointer to the full-resolution shared mask (single-channel).
 * @param h_output Host pointer to where the final blended images will be copied.
 * @param imageWidth Width of each full-resolution image.
 * @param imageHeight Height of each full-resolution image.
 * @param numLevels Number of pyramid levels.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
template <typename T, typename F_T = float>
cudaError_t cudaBatchedLaplacianBlend(
    const T* h_image1,
    const T* h_image2,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int numLevels,
    int batchSize,
    cudaStream_t stream);

/**
 * @brief Performs batched Laplacian blending using a preallocated context.
 *
 * This host function leverages a preallocated CudaBatchLaplacianBlendContext to store intermediate
 * pyramid arrays and parameters. It builds Gaussian and Laplacian pyramids for the image sets and mask,
 * blends the Laplacian pyramids, and reconstructs the final blended image directly into device memory.
 *
 * @tparam T The image data type.
 * @param d_image1 Device pointer to the first set of full-resolution images.
 * @param d_image2 Device pointer to the second set of full-resolution images.
 * @param d_mask Device pointer to the shared mask.
 * @param d_output Device pointer where the final blended images will be stored.
 * @param context Reference to a CudaBatchLaplacianBlendContext that holds preallocated buffers and blending parameters.
 * @param stream CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
template <typename T, typename F_T = float>
cudaError_t cudaBatchedLaplacianBlendWithContext(
    const T* d_image1,
    const T* d_image2,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext<T>& context,
    int channels,
    cudaStream_t stream);
