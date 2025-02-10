#include "cudaMat.h"
#include <cuda_runtime.h>
#include <cassert>

namespace hm {

#ifdef WITH_JETSON_UTILS
/**
 * @brief Converts an OpenCV cv::Mat to the corresponding jetson‑utils imageFormat.
 *
 * Inspects the cv::Mat’s depth and channel count to determine the appropriate format.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding jetson‑utils imageFormat.
 */
imageFormat cvMatToImageFormat(const cv::Mat& mat) {
  int depth = mat.depth(); // e.g. CV_8U, CV_32F, etc.
  int channels = mat.channels(); // e.g. 1, 3, or 4

  if (depth == CV_8U) {
    if (channels == 1)
      return IMAGE_GRAY8;
    else if (channels == 3)
      return IMAGE_BGR8; // (or IMAGE_RGB8 if channels are swapped)
    else if (channels == 4)
      return IMAGE_BGRA8;
  } else if (depth == CV_32F) {
    if (channels == 1)
      return IMAGE_GRAY32F;
    else if (channels == 3)
      return IMAGE_BGR32F;
    else if (channels == 4)
      return IMAGE_BGRA32F;
  }
  return IMAGE_UNKNOWN;
}

/**
 * @brief Converts a jetson‑utils imageFormat to an OpenCV type constant.
 *
 * Maps the provided imageFormat to an OpenCV type such as CV_8UC3.
 *
 * @param fmt The jetson‑utils image format.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int imageFormatToCvType(imageFormat fmt) {
  switch (fmt) {
    case IMAGE_GRAY8:
      return CV_8UC1;
    case IMAGE_BGR8:
      return CV_8UC3;
    case IMAGE_BGRA8:
      return CV_8UC4;
    case IMAGE_GRAY32F:
      return CV_32FC1;
    case IMAGE_BGR32F:
      return CV_32FC3;
    case IMAGE_BGRA32F:
      return CV_32FC4;
    default:
      return -1; // Unknown format
  }
}
#endif // WITH_JETSON_UTILS

/**
 * @brief Converts an OpenCV cv::Mat to a CudaPixelType.
 *
 * Determines the CUDA pixel type based on the cv::Mat’s depth and channel count.
 * For example, CV_8UC3 is mapped to CUDA_PIXEL_UCHAR3.
 * For half-precision data (CV_16F) the mapping is to CUDA_PIXEL_HALF*.
 * For 16-bit unsigned images (CV_16U) the mapping is to CUDA_PIXEL_USHORT*.
 *
 * @param mat The input cv::Mat.
 * @return The corresponding CudaPixelType.
 */
CudaPixelType cvMatToCudaPixelType(const cv::Mat& mat) {
  int depth = mat.depth();
  int channels = mat.channels();

  // Map 8-bit unsigned.
  if (depth == CV_8U) {
    if (channels == 1)
      return CUDA_PIXEL_UCHAR1;
    else if (channels == 3)
      return CUDA_PIXEL_UCHAR3;
    else if (channels == 4)
      return CUDA_PIXEL_UCHAR4;
  }
  // Map 16-bit unsigned.
  else if (depth == CV_16U) {
    if (channels == 1)
      return CUDA_PIXEL_USHORT1;
    else if (channels == 3)
      return CUDA_PIXEL_USHORT3;
    else if (channels == 4)
      return CUDA_PIXEL_USHORT4;
  }
  // Map 32-bit signed int.
  else if (depth == CV_32S) {
    if (channels == 1)
      return CUDA_PIXEL_INT1;
    else if (channels == 3)
      return CUDA_PIXEL_INT3;
    else if (channels == 4)
      return CUDA_PIXEL_INT4;
  }
  // Map 32-bit float.
  else if (depth == CV_32F) {
    if (channels == 1)
      return CUDA_PIXEL_FLOAT1;
    else if (channels == 3)
      return CUDA_PIXEL_FLOAT3;
    else if (channels == 4)
      return CUDA_PIXEL_FLOAT4;
  }
  // Map 16-bit float (half precision). OpenCV uses CV_16F.
  else if (depth == CV_16F) {
    if (channels == 1)
      return CUDA_PIXEL_HALF1;
    else if (channels == 3)
      return CUDA_PIXEL_HALF3;
    else if (channels == 4)
      return CUDA_PIXEL_HALF4;
  }
  // (Mapping for bfloat16 could be added if desired.)
  return CUDA_PIXEL_UNKNOWN;
}

/**
 * @brief Converts a CudaPixelType to an OpenCV type constant.
 *
 * Maps a CUDA pixel type (e.g. CUDA_PIXEL_UCHAR3) to its corresponding OpenCV type (e.g. CV_8UC3).
 *
 * @param fmt The CUDA pixel type.
 * @return The corresponding OpenCV type constant, or -1 if unknown.
 */
int cudaPixelTypeToCvType(CudaPixelType fmt) {
  switch (fmt) {
    case CUDA_PIXEL_UCHAR1:
      return CV_8UC1;
    case CUDA_PIXEL_UCHAR3:
      return CV_8UC3;
    case CUDA_PIXEL_UCHAR4:
      return CV_8UC4;
    case CUDA_PIXEL_USHORT1:
      return CV_16UC1;
    case CUDA_PIXEL_USHORT3:
      return CV_16UC3;
    case CUDA_PIXEL_USHORT4:
      return CV_16UC4;
    case CUDA_PIXEL_INT1:
      return CV_32SC1;
    case CUDA_PIXEL_INT3:
      return CV_32SC3;
    case CUDA_PIXEL_INT4:
      return CV_32SC4;
    case CUDA_PIXEL_FLOAT1:
      return CV_32FC1;
    case CUDA_PIXEL_FLOAT3:
      return CV_32FC3;
    case CUDA_PIXEL_FLOAT4:
      return CV_32FC4;
    case CUDA_PIXEL_HALF1:
      return CV_16FC1;
    case CUDA_PIXEL_HALF3:
      return CV_16FC3;
    case CUDA_PIXEL_HALF4:
      return CV_16FC4;
    // For bfloat16 we also use 16-bit float codes (even though representation differs)
    case CUDA_PIXEL_BF16_1:
      return CV_16FC1;
    case CUDA_PIXEL_BF16_3:
      return CV_16FC3;
    case CUDA_PIXEL_BF16_4:
      return CV_16FC4;
    default:
      return -1;
  }
}

/**
 * @brief Returns the element size in bytes for a given CUDA pixel type.
 *
 * For example, CUDA_PIXEL_UCHAR3 returns 3 bytes; CUDA_PIXEL_HALF4 returns 8 bytes.
 *
 * @param fmt The CUDA pixel type.
 * @return The size in bytes for one element, or 0 if unknown.
 */
size_t cudaPixelElementSize(CudaPixelType fmt) {
  switch (fmt) {
    case CUDA_PIXEL_UCHAR1:
      return 1;
    case CUDA_PIXEL_UCHAR3:
      return 3;
    case CUDA_PIXEL_UCHAR4:
      return 4;
    case CUDA_PIXEL_USHORT1:
      return 2;
    case CUDA_PIXEL_USHORT3:
      return 6;
    case CUDA_PIXEL_USHORT4:
      return 8;
    case CUDA_PIXEL_INT1:
      return 4;
    case CUDA_PIXEL_INT3:
      return 12;
    case CUDA_PIXEL_INT4:
      return 16;
    case CUDA_PIXEL_FLOAT1:
      return 4;
    case CUDA_PIXEL_FLOAT3:
      return 12;
    case CUDA_PIXEL_FLOAT4:
      return 16;
    case CUDA_PIXEL_HALF1:
      return 2;
    case CUDA_PIXEL_HALF3:
      return 6;
    case CUDA_PIXEL_HALF4:
      return 8;
    case CUDA_PIXEL_BF16_1:
      return 2;
    case CUDA_PIXEL_BF16_3:
      return 6;
    case CUDA_PIXEL_BF16_4:
      return 8;
    default:
      return 0;
  }
}
} // namespace hm
