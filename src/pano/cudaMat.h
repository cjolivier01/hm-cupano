#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>

#include "cudaTypes.h"

#ifdef WITH_JETSON_UTILS
#include "imageFormat.h" // Assumed to define the jetson‑utils imageFormat enum (e.g. IMAGE_BGR8, etc.)
#endif

namespace hm {
/**
 * @file cudaMat.h
 * @brief Utilities for converting between OpenCV cv::Mat, jetson‑utils image formats,
 *        and CUDA pixel types, plus a helper class for managing device memory.
 *
 * This header provides:
 *  - Functions to convert between cv::Mat and jetson‑utils imageFormat.
 *  - Functions to convert between cv::Mat and custom CUDA pixel types.
 *  - Template specializations to map CUDA pixel type enum values to actual CUDA types.
 *  - A template class CudaMat to allocate and manage device memory for one or more images.
 *  - A helper function to return the element size (in bytes) of a given CUDA pixel type.
 */

/*----------------------------------------------------------------------------
  Non‑template function declarations (implemented in cudaMat.cpp)
-----------------------------------------------------------------------------*/
#ifdef WITH_JETSON_UTILS
/**
 * @brief Converts an OpenCV cv::Mat to the corresponding jetson‑utils imageFormat.
 *
 * Inspects the cv::Mat’s depth and channel count to determine the format.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding jetson‑utils imageFormat.
 */
imageFormat cvMatToImageFormat(const cv::Mat& mat);

/**
 * @brief Converts a jetson‑utils imageFormat to an OpenCV type constant.
 *
 * Maps formats (e.g. IMAGE_BGR8) to their corresponding OpenCV type (e.g. CV_8UC3).
 *
 * @param fmt The jetson‑utils image format.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int imageFormatToCvType(imageFormat fmt);
#endif // WITH_JETSON_UTILS

/**
 * @brief Enumeration of CUDA pixel types.
 *
 * This enum defines common CUDA pixel types—including support for 8‑bit,
 * 32‑bit, 16‑bit float (half), 16‑bit unsigned short, and bfloat16 types.
 */
enum CudaPixelType {
  CUDA_PIXEL_UNKNOWN = -1, ///< Unknown pixel type.
  CUDA_PIXEL_UCHAR1, ///< 8-bit, 1 channel (unsigned char).
  CUDA_PIXEL_UCHAR3, ///< 8-bit, 3 channels (uchar3).
  CUDA_PIXEL_UCHAR4, ///< 8-bit, 4 channels (uchar4).
  CUDA_PIXEL_USHORT1, ///< 16-bit unsigned, 1 channel (unsigned short).
  CUDA_PIXEL_USHORT3, ///< 16-bit unsigned, 3 channels (ushort3).
  CUDA_PIXEL_USHORT4, ///< 16-bit unsigned, 4 channels (ushort4).
  CUDA_PIXEL_INT1, ///< 32-bit int, 1 channel (int).
  CUDA_PIXEL_INT3, ///< 32-bit int, 3 channels (int3).
  CUDA_PIXEL_INT4, ///< 32-bit int, 4 channels (int4).
  CUDA_PIXEL_FLOAT1, ///< 32-bit float, 1 channel (float).
  CUDA_PIXEL_FLOAT3, ///< 32-bit float, 3 channels (float3).
  CUDA_PIXEL_FLOAT4, ///< 32-bit float, 4 channels (float4).
  CUDA_PIXEL_HALF1, ///< 16-bit float (half), 1 channel (__half).
  CUDA_PIXEL_HALF3, ///< 16-bit float (half), 3 channels (half3).
  CUDA_PIXEL_HALF4, ///< 16-bit float (half), 4 channels (custom half4).
  CUDA_PIXEL_BF16_1, ///< 16-bit bfloat, 1 channel (__nv_bfloat16).
  CUDA_PIXEL_BF16_3, ///< 16-bit bfloat, 3 channels (bfloat16_3).
  CUDA_PIXEL_BF16_4 ///< 16-bit bfloat, 4 channels (bfloat16_4).
};

/**
 * @brief Converts an OpenCV cv::Mat to a CudaPixelType.
 *
 * Determines the CUDA pixel type based on the cv::Mat’s depth and channel count.
 * For example, CV_8UC3 is mapped to CUDA_PIXEL_UCHAR3. If the cv::Mat
 * has a depth corresponding to 16-bit floating point (CV_16F) then it is mapped to a half type,
 * and if it is CV_16U then it is mapped to a USHORT type.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding CudaPixelType.
 */
CudaPixelType cvMatToCudaPixelType(const cv::Mat& mat);

/**
 * @brief Converts a CudaPixelType to an OpenCV type constant.
 *
 * Maps a CUDA pixel type (e.g. CUDA_PIXEL_UCHAR3) to its corresponding OpenCV type (e.g. CV_8UC3).
 *
 * @param fmt The CUDA pixel type.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int cudaPixelTypeToCvType(CudaPixelType fmt);

/**
 * @brief Returns the element size in bytes for a given CUDA pixel type.
 *
 * For example, CUDA_PIXEL_UCHAR3 returns 3 bytes; CUDA_PIXEL_HALF4 returns 8 bytes.
 *
 * @param fmt The CUDA pixel type.
 * @return The size in bytes for one element, or 0 if unknown.
 */
size_t cudaPixelElementSize(CudaPixelType fmt);

/*----------------------------------------------------------------------------
  Template Specializations: Mapping from a CudaPixelType enumerator to the corresponding CUDA type.
-----------------------------------------------------------------------------*/

/**
 * @brief Template mapping from a CudaPixelType enumeration to the corresponding CUDA type.
 *
 * Specializations define a nested alias `type` corresponding to the actual CUDA vector or scalar type.
 *
 * @tparam T The enumerator from CudaPixelType.
 */
template <CudaPixelType T>
struct CudaPixelTypeToCudaType; // Primary template declaration (no definition).

// Specializations:

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR1> {
  using type = unsigned char;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR3> {
  using type = uchar3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_UCHAR4> {
  using type = uchar4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_USHORT1> {
  using type = unsigned short;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_USHORT3> {
  using type = ushort3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_USHORT4> {
  using type = ushort4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT1> {
  using type = int;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT3> {
  using type = int3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_INT4> {
  using type = int4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT1> {
  using type = float;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT3> {
  using type = float3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_FLOAT4> {
  using type = float4;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_HALF1> {
  using type = __half;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_HALF3> {
  using type = half3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_HALF4> {
  using type = half4; // Our custom half4.
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_BF16_1> {
  using type = __nv_bfloat16;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_BF16_3> {
  using type = bfloat16_3;
};

template <>
struct CudaPixelTypeToCudaType<CUDA_PIXEL_BF16_4> {
  using type = bfloat16_4;
};

//
// Primary template declaration – no definition is provided.
//
template <typename T>
struct CudaTypeToPixelType;

// --- 1-channel types ---
template <>
struct CudaTypeToPixelType<unsigned char> {
  static constexpr CudaPixelType value = CUDA_PIXEL_UCHAR1;
};

template <>
struct CudaTypeToPixelType<uchar1> {
  static constexpr CudaPixelType value = CUDA_PIXEL_UCHAR1;
};

template <>
struct CudaTypeToPixelType<unsigned short> {
  static constexpr CudaPixelType value = CUDA_PIXEL_USHORT1;
};

template <>
struct CudaTypeToPixelType<int> {
  static constexpr CudaPixelType value = CUDA_PIXEL_INT1;
};

template <>
struct CudaTypeToPixelType<float> {
  static constexpr CudaPixelType value = CUDA_PIXEL_FLOAT1;
};

template <>
struct CudaTypeToPixelType<__half> {
  static constexpr CudaPixelType value = CUDA_PIXEL_HALF1;
};

template <>
struct CudaTypeToPixelType<__nv_bfloat16> {
  static constexpr CudaPixelType value = CUDA_PIXEL_BF16_1;
};

// --- 3-channel types ---
template <>
struct CudaTypeToPixelType<uchar3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_UCHAR3;
};

template <>
struct CudaTypeToPixelType<ushort3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_USHORT3;
};

template <>
struct CudaTypeToPixelType<int3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_INT3;
};

template <>
struct CudaTypeToPixelType<float3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_FLOAT3;
};

template <>
struct CudaTypeToPixelType<half3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_HALF3;
};

template <>
struct CudaTypeToPixelType<bfloat16_3> {
  static constexpr CudaPixelType value = CUDA_PIXEL_BF16_3;
};

// --- 4-channel types ---
template <>
struct CudaTypeToPixelType<uchar4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_UCHAR4;
};

template <>
struct CudaTypeToPixelType<ushort4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_USHORT4;
};

template <>
struct CudaTypeToPixelType<int4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_INT4;
};

template <>
struct CudaTypeToPixelType<float4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_FLOAT4;
};

template <>
struct CudaTypeToPixelType<half4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_HALF4;
};

template <>
struct CudaTypeToPixelType<bfloat16_4> {
  static constexpr CudaPixelType value = CUDA_PIXEL_BF16_4;
};

/*----------------------------------------------------------------------------
  Template Class: CudaMat
-----------------------------------------------------------------------------*/

/**
 * @brief Templated class to manage CUDA device memory for one or more images.
 *
 * CudaMat allocates device memory for an image (or a batch of images) and provides functionality
 * to download the device memory back to a cv::Mat.
 *
 * The internal image type is now stored as a CudaPixelType.
 *
 * @tparam T The CUDA pixel type stored in device memory (default is float3).
 */
template <typename T = float3>
class CudaMat {
 public:
  // Delete copy and move constructors.
  CudaMat(const CudaMat&) = delete;
  CudaMat(CudaMat&&) = delete;

  /**
   * @brief Constructs a CudaMat from a single cv::Mat.
   *
   * Allocates device memory for the image and (optionally) copies the cv::Mat data to the device.
   *
   * @param mat The input cv::Mat.
   * @param copy If true, the data is copied to device memory.
   */
  CudaMat(const cv::Mat& mat, bool copy = true);

  /**
   * @brief Constructs a CudaMat from a batch of cv::Mat images.
   *
   * Allocates device memory for all images in the batch and (optionally) copies the data.
   *
   * @param mat_batch A vector of cv::Mat images.
   * @param copy If true, the data is copied to device memory.
   */
  CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy = true);

  /**
   * @brief Constructs a CudaMat with explicit dimensions and pixel type.
   *
   * Allocates device memory for a batch of images with dimensions B×W×H.
   * The provided channel count C must match the expected channel count for the given pixel type.
   *
   * @param B Batch size.
   * @param W Image width.
   * @param H Image height.
   * @param C Number of channels.
   * @param type The CUDA pixel type.
   */
  CudaMat(int B, int W, int H, int C, CudaPixelType type);

  CudaMat(int B, int W, int H, int C = 1);

  CudaMat(void* dataptr, int B, int W, int H, int C = 1);

  /**
   * @brief Destructor.
   *
   * Frees the allocated device memory.
   */
  ~CudaMat();

  /**
   * @brief Check whether the CudaMat is a valid surface
   *
   * @returns true if this surface is value (has memory allocated)
   */
  constexpr bool is_valid() const {
    return d_data_ != nullptr;
  }

  /**
   * @brief Downloads an image from device memory to a cv::Mat.
   *
   * For batched images, the parameter `batch_item` specifies which image to download.
   *
   * @param batch_item The index of the image to download (default is 0).
   * @return A cv::Mat containing the downloaded image.
   */
  cv::Mat download(int batch_item = 0) const;

  /// @brief Returns a pointer to the device memory.
  T* data();
  /// @brief Returns a const pointer to the device memory.
  const T* data() const;
  /// @brief Returns the image width (number of columns).
  constexpr int width() const;
  /// @brief Returns the image height (number of rows).
  constexpr int height() const;
  /// @brief Returns the OpenCV type of the image (converted from CudaPixelType).
  constexpr int type() const;
  /// @brief Returns the number of images in the batch.
  constexpr int batch_size() const;

  /**
   * @brief Returns a pointer to the raw underlying data.
   *
   * This returns a pointer of type BaseScalar_t<T>* which points to the underlying
   * base scalar data. For example, if T is float3, this returns a pointer of type float*.
   *
   * @return A pointer to the raw base scalar data.
   */
  BaseScalar_t<T>* data_raw();

  /**
   * @brief Returns a const pointer to the raw underlying data.
   *
   * @return A const pointer to the raw base scalar data.
   */
  const BaseScalar_t<T>* data_raw() const;

 private:
  T* d_data_{nullptr}; ///< Pointer to device memory.
  size_t size{0}; ///< Total size (in bytes) allocated on the device.
  int rows_{0}, cols_{0}; ///< Image dimensions.
  CudaPixelType type_{CUDA_PIXEL_UNKNOWN}; ///< CUDA pixel type for the image.
  int batch_size_{0}; ///< Number of images in the batch.
  bool owns_{true};
};
} // namespace hm

// Include inline implementations for template methods.
#include "cudaMat.inl"
