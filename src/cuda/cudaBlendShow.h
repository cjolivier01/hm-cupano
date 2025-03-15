#pragma once

#include "cudaBlend.h"

#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

namespace {
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
// Helper to determine an OpenCV type from the pixel type T and number of channels.
// (This version works for uchar (8-bit) or float pixel data.)
template <typename T>
inline int getCVTypeForPixel(int channels) {
  // Here we use sizeof(T) to decide if T is an 8-bit or 32-bit type.
  // For example, if T is uchar3 then sizeof(T)==3 and we assume 8-bit 3-channel.
  // If T is float3 then sizeof(T)==12 and we assume 32-bit 3-channel.
  if (sizeof(T) == 1 || sizeof(T) == 3) {
    if (channels == 1)
      return CV_8UC1;
    else if (channels == 3)
      return CV_8UC3;
    else if (channels == 4)
      return CV_8UC4;
  } else {
    if (channels == 1)
      return CV_32FC1;
    else if (channels == 3)
      return CV_32FC3;
    else if (channels == 4)
      return CV_32FC4;
  }
  return -1; // error
}

inline cv::Mat cloned_convert_to_uchar(const cv::Mat& image) {
  if (image.depth() == CV_32F || image.depth() == CV_64F) {
    cv::Mat img2;
    image.convertTo(img2, CV_8U);
    return img2 * 255;
  }
  // For non-floating point images, return a copy (or handle as needed)
  return image.clone();
}

template <typename T>
inline void displayPyramid(
    const std::string& windowName,
    const std::vector<T*>& pyramid,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    int channels,
    float scale) {
  const int cvType = getCVTypeForPixel<T>(channels);
  int totalHeight = 0;
  int maxWidth = 0;
  std::vector<cv::Mat> levelMats;

  const int numLevels = pyramid.size();

  levelMats.reserve(numLevels);

  cudaDeviceSynchronize();
  // For each pyramid level, copy data from device and create a cv::Mat.
  for (int level = 0; level < numLevels; level++) {
    int w = widths[level];
    int h = heights[level];
    maxWidth = std::max(maxWidth, w);
    totalHeight += h;

    assert(pyramid[level]);

    std::unique_ptr<T[]> buffer = std::make_unique<T[]>(w * h);
    const size_t dataSize = w * h * sizeof(T) * channels;
    cudaError_t cuerr = cudaMemcpy(buffer.get(), pyramid[level], dataSize, cudaMemcpyDeviceToHost);
    assert(cuerr == cudaError_t::cudaSuccess);

    // Create a cv::Mat header that points to the host data.
    // (We clone immediately so that the Mat owns its data.)
    cv::Mat levelMat = cv::Mat(h, w, cvType, buffer.get()).clone();

    cv::imshow(windowName, levelMat);
    cv::waitKey(0);

    levelMats.emplace_back(cloned_convert_to_uchar(levelMat));
  }

  // Create a composite image that is large enough to hold all levels stacked vertically.
  cv::Mat composite(totalHeight, maxWidth, cvType, cv::Scalar::all(0));

  int currentY = 0;
  for (const auto& mat : levelMats) {
    // Create a region-of-interest (ROI) in the composite image.
    cv::Rect roi(0, currentY, mat.cols, mat.rows);
    // Copy the pyramid level image into the composite image.
    mat.copyTo(composite(roi));
    currentY += mat.rows;
  }

  if (scale != 1.0f) {
    // Calculate new dimensions
    int newWidth = static_cast<int>(composite.cols * scale);
    int newHeight = static_cast<int>(composite.rows * scale);

    // Resize the image
    cv::resize(composite, composite, cv::Size(newWidth, newHeight));
  }

  // Display the composite image.
  cv::imshow(windowName, composite);
}

} // namespace

template <typename T>
inline void CudaBatchLaplacianBlendContext<T>::displayPyramids(int channels, float scale) const {
  // Determine the OpenCV type from T and the number of channels.
  const int cvType = getCVTypeForPixel<T>(channels);
  if (cvType == -1) {
    printf("Unsupported pixel type or channel count.\n");
    return;
  }

  // Display different pyramids. (Adjust which ones you want to show.)
  // displayPyramid("Gaussian 1", d_gauss1, channels);
  // displayPyramid("Gaussian 2", d_gauss2, channels);
  displayPyramid("Mask Pyramid", d_maskPyr, widths, heights, 1, scale); // assuming mask is single channel
  // displayPyramid("Laplacian 1", d_lap1, channels);
  // displayPyramid("Laplacian 2", d_lap2, channels);
  // displayPyramid("Blended Pyramid", d_blend, channels);
  // Optionally, you could also display the reconstructed images from d_resonstruct if desired.

  // Wait for a key press to close the windows.
  cv::waitKey(0);
}
