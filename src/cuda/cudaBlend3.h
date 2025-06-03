// cudaBlend3.h
#pragma once

/**
 * @file cudaBlend3.h
 * @brief CUDA-accelerated batched Laplacian blending for **three** images.
 *
 * This header defines the types, host functions, and templated CUDA kernels for
 * performing batched Laplacian blending on **three** input image sets using a single
 * 3-channel mask. The blending process builds Gaussian and Laplacian pyramids for
 * three input image sets and a shared mask (with three channels), blends the
 * Laplacian pyramids via per-pixel weights (m₁,m₂,m₃), and then reconstructs the
 * final blended images.
 *
 * The implementation supports different image data types (e.g. float, unsigned char,
 * __half, __nv_bfloat16) and uses CUDA streams for asynchronous execution.
 */

#include "src/pano/cudaMat.h"
#include "src/utils/showImage.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

/**
 * @brief Context structure for batched Laplacian blending on **three** images.
 *
 * This templated structure stores the parameters and device pointers required for
 * batched Laplacian blending of three image sets. It maintains device memory for
 * Gaussian and Laplacian pyramid levels, as well as for intermediate blend and
 * reconstruction results.
 *
 * Note that the level-0 device pointers (for the original full-resolution images
 * and 3-channel mask) are expected to be managed externally (by the caller) and
 * are not freed in the destructor.
 *
 * @tparam T The data type for the images (e.g. float, unsigned char, __half, __nv_bfloat16).
 */
template <typename T>
struct CudaBatchLaplacianBlendContext3 {
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
  CudaBatchLaplacianBlendContext3(int image_width, int image_height, int num_levels, int batch_size)
      : numLevels(num_levels),
        imageWidth(image_width),
        imageHeight(image_height),
        batchSize(batch_size),
        widths(num_levels),
        heights(num_levels),
        d_gauss1(num_levels, nullptr),
        d_gauss2(num_levels, nullptr),
        d_gauss3(num_levels, nullptr),
        d_maskPyr(num_levels, nullptr),
        d_lap1(num_levels, nullptr),
        d_lap2(num_levels, nullptr),
        d_lap3(num_levels, nullptr),
        d_blend(num_levels, nullptr),
        d_reconstruct(num_levels, nullptr) {}

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
   * Laplacian, blend, and reconstruction buffers. For levels > 0, the Gaussian and mask pyramid
   * arrays are also freed.
   */
  ~CudaBatchLaplacianBlendContext3() {
    for (int level = 0; level < numLevels; level++) {
      maybeCudaFree(d_lap1[level]);
      maybeCudaFree(d_lap2[level]);
      maybeCudaFree(d_lap3[level]);
      maybeCudaFree(d_blend[level]);
      if (level) {
        // Level 0 pointers are owned by user code (passed into the function each time)
        maybeCudaFree(d_gauss1[level]);
        maybeCudaFree(d_gauss2[level]);
        maybeCudaFree(d_gauss3[level]);
        maybeCudaFree(d_maskPyr[level]);
        maybeCudaFree(d_reconstruct[level]);
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

  // Gaussian pyramids for three image sets:
  std::vector<T*> d_gauss1; ///< Device pointers for Gaussian pyramid (image set 1).
  std::vector<T*> d_gauss2; ///< Device pointers for Gaussian pyramid (image set 2).
  std::vector<T*> d_gauss3; ///< Device pointers for Gaussian pyramid (image set 3).

  std::vector<T*> d_maskPyr; ///< Device pointers for the downsampled 3-channel shared mask.

  // Laplacian pyramids for three image sets:
  std::vector<T*> d_lap1; ///< Device pointers for Laplacian pyramid (image 1).
  std::vector<T*> d_lap2; ///< Device pointers for Laplacian pyramid (image 2).
  std::vector<T*> d_lap3; ///< Device pointers for Laplacian pyramid (image 3).

  std::vector<T*> d_blend; ///< Device pointers for the blended Laplacian pyramid.
  std::vector<T*> d_reconstruct; ///< Device pointers for temporary reconstruction buffers.

  bool initialized{false}; ///< Flag indicating whether the context has been initialized.

  // Must link to utils for this
  inline void displayPyramids(int channels, float scale, bool wait) const;

  void show_image(const std::string& label, std::vector<T*>& vec_d_ptrs, int level, int channels, bool wait) {
    T* d_ptr = vec_d_ptrs.at(level);
    // CudaMat(T* dataptr, int B, int W, int H, int C = 1);
    hm::CudaMat<T> mat(d_ptr, 1, widths.at(level), heights.at(level), channels);
    hm::utils::show_image(label, mat.download(), wait);
  }
};

/**
 * @brief Performs batched Laplacian blending on **three** images.
 *
 * This host function copies the input full-resolution images and shared 3-channel mask
 * from host to device, builds Gaussian and Laplacian pyramids for all three image sets,
 * blends the Laplacian pyramids using the 3-channel mask, reconstructs the final blended
 * images, and copies the result back to host memory.
 *
 * @tparam T The image data type.
 * @tparam F_T The floating-point type used for intermediate computations (default = float).
 * @param h_image1 Host pointer to the first set of full-resolution images (batched layout).
 * @param h_image2 Host pointer to the second set of full-resolution images (batched layout).
 * @param h_image3 Host pointer to the third set of full-resolution images (batched layout).
 * @param h_mask   Host pointer to the full-resolution **3-channel** shared mask.
 *                 Layout: [H × W × 3], where each pixel has (m₁,m₂,m₃).
 * @param h_output Host pointer to where the final blended images will be copied.
 * @param imageWidth Width of each full-resolution image.
 * @param imageHeight Height of each full-resolution image.
 * @param channels Number of image channels (e.g. 3 for RGB, 4 for RGBA).
 * @param numLevels Number of pyramid levels.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
template <typename T, typename F_T = float>
cudaError_t cudaBatchedLaplacianBlend3(
    const T* h_image1,
    const T* h_image2,
    const T* h_image3,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int numLevels,
    int batchSize,
    cudaStream_t stream);

/**
 * @brief Performs batched Laplacian blending using a preallocated context for **three** images.
 *
 * This host function leverages a preallocated CudaBatchLaplacianBlendContext3 to store intermediate
 * pyramid arrays and parameters. It builds Gaussian and Laplacian pyramids for the three image sets
 * and the 3-channel mask, blends the Laplacian pyramids, and reconstructs the final blended image
 * directly into device memory.
 *
 * @tparam T The image data type.
 * @tparam F_T The floating-point type used for intermediate computations (default = float).
 * @param d_image1 Device pointer to the first set of full-resolution images.
 * @param d_image2 Device pointer to the second set of full-resolution images.
 * @param d_image3 Device pointer to the third set of full-resolution images.
 * @param d_mask   Device pointer to the shared **3-channel** mask.
 * @param d_output Device pointer where the final blended images will be stored.
 * @param context  Reference to a CudaBatchLaplacianBlendContext3 that holds preallocated buffers and blending
 * parameters.
 * @param channels Number of image channels (e.g. 3 for RGB, 4 for RGBA).
 * @param stream   CUDA stream to use for all kernel launches and memory copies (default is 0).
 * @return cudaError_t CUDA error code.
 */
template <typename T, typename F_T = float>
cudaError_t cudaBatchedLaplacianBlendWithContext3(
    const T* d_image1,
    const T* d_image2,
    const T* d_image3,
    const uint16_t* map1_x,
    const uint16_t* map1_y,
    const uint16_t* map2_x,
    const uint16_t* map2_y,
    const uint16_t* map3_x,
    const uint16_t* map3_y,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext3<T>& context,
    int channels,
    cudaStream_t stream);

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
      int numLevels,                                         \
      int batchSize,                                         \
      cudaStream_t stream);

#define INSTANTIATE_CUDA_BATCHED_LAPLACIAN_BLEND_WITH_CONTEXT3(T)       \
  template cudaError_t cudaBatchedLaplacianBlendWithContext3<T, float>( \
      const T* d_image1,                                                \
      const T* d_image2,                                                \
      const T* d_image3,                                                \
      const uint16_t* map1_x,                                           \
      const uint16_t* map1_y,                                           \
      const uint16_t* map2_x,                                           \
      const uint16_t* map2_y,                                           \
      const uint16_t* map3_x,                                           \
      const uint16_t* map3_y,                                           \
      const T* d_mask,                                                  \
      T* d_output,                                                      \
      CudaBatchLaplacianBlendContext3<T>& context,                      \
      int channels,                                                     \
      cudaStream_t stream);
