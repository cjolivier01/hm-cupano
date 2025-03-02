/**
 * @file cudaBlend.h
 * @brief Batched Laplacian blending for images using CUDA.
 *
 * This file contains the declarations, definitions, and templated kernels for
 * performing batched Laplacian blending. The blending is performed by building
 * Gaussian and Laplacian pyramids for two image sets and a shared mask, blending
 * the Laplacian pyramids, and reconstructing the final blended image.
 */

#pragma once

#include <cuda_runtime.h>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>

/**
 * @brief Batched Context Structure for Laplacian Blending (images only).
 *
 * This structure holds all the device pointers and parameters needed for performing
 * batched Laplacian blending. The mask is shared across the batch.
 *
 * @tparam T The data type for the images (e.g. float, unsigned char, __half, __nv_bfloat16).
 */
template <typename T>
struct CudaBatchLaplacianBlendContext {
  /**
   * @brief Constructor.
   *
   * Initializes the context with the image dimensions, number of pyramid levels,
   * and batch size. The device pointer vectors are sized appropriately.
   *
   * @param image_width Width of the full-resolution image.
   * @param image_height Height of the full-resolution image.
   * @param num_levels Number of pyramid levels.
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
   * @brief Helper to free a CUDA pointer if non-null.
   *
   * @param p Pointer to free.
   */
  static constexpr void maybeCudaFree(void* p) {
    if (p) {
      cudaFree(p);
    }
  }

  /**
   * @brief Destructor.
   *
   * Frees all device memory allocations (except level 0, which is assumed to be
   * managed by user code).
   */
  ~CudaBatchLaplacianBlendContext() {
    for (int level = 0; level < numLevels; level++) {
      maybeCudaFree(d_lap1[level]);
      maybeCudaFree(d_lap2[level]);
      maybeCudaFree(d_blend[level]);
      if (level) {
        // Level 0 is owned by user code (passed into the function each time)
        maybeCudaFree(d_gauss1[level]);
        maybeCudaFree(d_gauss2[level]);
        maybeCudaFree(d_maskPyr[level]);
        maybeCudaFree(d_resonstruct[level]);
      }
    }
  }

  const int numLevels;           ///< Number of pyramid levels.
  const int imageWidth;          ///< Width of the full-resolution image.
  const int imageHeight;         ///< Height of the full-resolution image.
  const int batchSize;           ///< Number of images in the batch.
  size_t allocation_size{0};     ///< Total allocated device memory size.
  std::vector<int> widths;       ///< Widths of images at each pyramid level.
  std::vector<int> heights;      ///< Heights of images at each pyramid level.
  std::vector<T*> d_gauss1;      ///< Gaussian pyramid for first image.
  std::vector<T*> d_gauss2;      ///< Gaussian pyramid for second image.
  std::vector<T*> d_maskPyr;     ///< Pyramid for the shared mask.
  std::vector<T*> d_lap1;        ///< Laplacian pyramid for first image.
  std::vector<T*> d_lap2;        ///< Laplacian pyramid for second image.
  std::vector<T*> d_blend;       ///< Blended Laplacian pyramid.
  std::vector<T*> d_resonstruct; ///< Temporary arrays for reconstruction.
  bool initialized{false};       ///< Flag indicating if the context has been initialized.
};

/**
 * @brief Batched Laplacian blending.
 *
 * Copies host images (batched layout) and a shared mask to device memory, builds Gaussian and Laplacian pyramids,
 * blends the Laplacian pyramids, reconstructs the final blended images, and copies the result back to host.
 *
 * @tparam T The image data type.
 * @param h_image1 Host pointer to the first set of full-resolution images.
 * @param h_image2 Host pointer to the second set of full-resolution images.
 * @param h_mask Host pointer to the full-resolution shared mask.
 * @param h_output Host pointer where the final blended images will be copied.
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
    int numLevels,
    int batchSize,
    cudaStream_t stream);

/**
 * @brief Batched Laplacian blending with a preallocated context.
 *
 * Uses a preallocated context to store intermediate pyramid arrays, builds Gaussian and Laplacian pyramids,
 * blends the Laplacian pyramids, reconstructs the final blended image, and stores the result in d_output.
 *
 * @tparam T The image data type.
 * @param d_image1 Device pointer to the first set of full-resolution images.
 * @param d_image2 Device pointer to the second set of full-resolution images.
 * @param d_mask Device pointer to the shared mask.
 * @param d_output Device pointer where the final blended images will be stored.
 * @param context Reference to a CudaBatchLaplacianBlendContext structure that holds preallocated arrays and blending
 * parameters.
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
    cudaStream_t stream);
