#pragma once

#include <cuda_runtime.h>
#include <cassert>
#include "cudaMat.h"

namespace hm {

// Helper: Returns the number of channels expected for a given CUDA pixel type.
static inline int cudaPixelTypeChannels(CudaPixelType fmt) {
  switch (fmt) {
    case CUDA_PIXEL_UCHAR1:
      return 1;
    case CUDA_PIXEL_UCHAR3:
      return 3;
    case CUDA_PIXEL_UCHAR4:
      return 4;
    case CUDA_PIXEL_USHORT1:
      return 1;
    case CUDA_PIXEL_USHORT3:
      return 3;
    case CUDA_PIXEL_USHORT4:
      return 4;
    case CUDA_PIXEL_INT1:
      return 1;
    case CUDA_PIXEL_INT3:
      return 3;
    case CUDA_PIXEL_INT4:
      return 4;
    case CUDA_PIXEL_FLOAT1:
      return 1;
    case CUDA_PIXEL_FLOAT3:
      return 3;
    case CUDA_PIXEL_FLOAT4:
      return 4;
    case CUDA_PIXEL_HALF1:
      return 1;
    case CUDA_PIXEL_HALF3:
      return 3;
    case CUDA_PIXEL_HALF4:
      return 4;
    case CUDA_PIXEL_BF16_1:
      return 1;
    case CUDA_PIXEL_BF16_3:
      return 3;
    case CUDA_PIXEL_BF16_4:
      return 4;
    default:
      return 0;
  }
}

/**
 * @brief Constructs a CudaMat from a single cv::Mat.
 *
 * Allocates device memory and (optionally) copies the cv::Mat data to the device.
 *
 * @tparam T The CUDA pixel type.
 * @param mat The input cv::Mat.
 * @param copy If true, copies the data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const cv::Mat& mat, bool copy) : rows_(mat.rows), cols_(mat.cols), batch_size_(1) {
  // Convert the cv::Mat type (an int) to a CudaPixelType.
  type_ = cvMatToCudaPixelType(mat);
  size_t expectedElemSize = cudaPixelElementSize(type_);
  // Verify that the cv::Mat's element size matches what we expect.
  assert(mat.elemSize() == expectedElemSize);
  size = mat.total() * expectedElemSize;
  cudaMalloc(&d_data_, size);
  assert(mat.isContinuous());
  if (copy) {
    cudaMemcpy(d_data_, mat.data, size, cudaMemcpyHostToDevice);
  }
}

/**
 * @brief Constructs a CudaMat from a batch of cv::Mat images.
 *
 * Allocates device memory to hold all images and (optionally) copies each image to the device.
 *
 * @tparam T The CUDA pixel type.
 * @param mat_batch Vector of input cv::Mat images.
 * @param copy If true, copies data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy)
    : batch_size_(static_cast<int>(mat_batch.size())) {
  assert(batch_size_ > 0);
  const cv::Mat& first = mat_batch.at(0);
  rows_ = first.rows;
  cols_ = first.cols;
  type_ = cvMatToCudaPixelType(first);
  size_t expectedElemSize = cudaPixelElementSize(type_);
  assert(first.elemSize() == expectedElemSize);
  size_t size_each = first.total() * expectedElemSize;
  size = size_each * batch_size_;
  cudaError_t cuerr = cudaMalloc(&d_data_, size);
  if (cuerr == cudaError_t::cudaSuccess && copy) {
    uint8_t* p = reinterpret_cast<uint8_t*>(d_data_);
    for (const cv::Mat& mat : mat_batch) {
      assert(mat.isContinuous());
      // It is assumed that each mat in the batch has the same dimensions and type.
      assert(mat.rows == rows_ && mat.cols == cols_);
      assert(mat.elemSize() == expectedElemSize);
      cudaMemcpy(p, mat.data, size_each, cudaMemcpyHostToDevice);
      p += size_each;
    }
  }
}

/**
 * @brief Constructs a CudaMat with explicit dimensions and pixel type.
 *
 * Allocates device memory for a batch of images with dimensions B×W×H.
 * The provided channel count C must match the expected channel count for the given pixel type.
 *
 * @tparam T The CUDA pixel type.
 * @param B Batch size.
 * @param W Image width.
 * @param H Image height.
 * @param C Number of channels.
 * @param type The CUDA pixel type.
 */
template <typename T>
CudaMat<T>::CudaMat(int B, int W, int H, int C, CudaPixelType type) : batch_size_(B), rows_(H), cols_(W), type_(type) {
  int expectedChannels = cudaPixelTypeChannels(type_);
  assert(expectedChannels == C);
  size_t elemSize = cudaPixelElementSize(type_);
  // Ensure that the template type T matches the expected element size.
  assert(sizeof(T) == elemSize);
  size = static_cast<size_t>(B * W * H) * elemSize;
  cudaMalloc(&d_data_, size);
}

template <typename T>
CudaMat<T>::CudaMat(int B, int W, int H, int C)
    : batch_size_(B),
      rows_(H),
      cols_(W),
      type_(CudaTypeToPixelType<T>::value) // automatically inferred from T
{
  assert(cudaPixelTypeChannels(type_) == C * sizeof(T) / sizeof(typename BaseScalar<T>::type));
  size_t elemSize = cudaPixelElementSize(type_);
  assert(sizeof(T) == elemSize);
  size = static_cast<size_t>(B * W * H) * elemSize;
  cudaMalloc(&d_data_, size);
}

template <typename T>
CudaMat<T>::CudaMat(void* d_data, int B, int W, int H, int C)
    : d_data_(static_cast<T*>(d_data)),
      batch_size_(B),
      rows_(H),
      cols_(W),
      type_(CudaTypeToPixelType<T>::value), // automatically inferred from T
      owns_(false) {
  assert(cudaPixelTypeChannels(type_) == C * sizeof(T) / sizeof(typename BaseScalar<T>::type));
  assert(sizeof(T) == cudaPixelElementSize(type_));
}

/**
 * @brief Destructor for CudaMat.
 *
 * Frees the allocated device memory.
 *
 * @tparam T The CUDA pixel type.
 */
template <typename T>
CudaMat<T>::~CudaMat() {
  if (d_data_ && owns_) {
    cudaFree(d_data_);
  }
}

/**
 * @brief Downloads an image from device memory to a cv::Mat.
 *
 * Downloads the image at the specified batch index.
 *
 * @tparam T The CUDA pixel type.
 * @param batch_item The index of the image in the batch (default is 0).
 * @return A cv::Mat containing the downloaded image.
 */
template <typename T>
cv::Mat CudaMat<T>::download(int batch_item) const {
  assert(batch_item >= 0 && batch_item < batch_size_);
  // Convert our stored CudaPixelType to an OpenCV type.
  int cvType = cudaPixelTypeToCvType(type_);
  cv::Mat mat(rows_, cols_, cvType);
  size_t elemSize = cudaPixelElementSize(type_);
  size_t size_each = static_cast<size_t>(rows_ * cols_) * elemSize;
  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_data_) + batch_item * size_each;
  cudaMemcpy(mat.data, src_ptr, size_each, cudaMemcpyDeviceToHost);
  return mat;
}

/**
 * @brief Returns a pointer to the device memory.
 *
 * @tparam T The CUDA pixel type.
 * @return Pointer to device memory.
 */
template <typename T>
T* CudaMat<T>::data() {
  return d_data_;
}

/**
 * @brief Returns a const pointer to the device memory.
 *
 * @tparam T The CUDA pixel type.
 * @return Const pointer to device memory.
 */
template <typename T>
const T* CudaMat<T>::data() const {
  return d_data_;
}

/**
 * @brief Returns the image width (number of columns).
 *
 * @tparam T The CUDA pixel type.
 * @return Image width.
 */
template <typename T>
constexpr int CudaMat<T>::width() const {
  return cols_;
}

/**
 * @brief Returns the image height (number of rows).
 *
 * @tparam T The CUDA pixel type.
 * @return Image height.
 */
template <typename T>
constexpr int CudaMat<T>::height() const {
  return rows_;
}

/**
 * @brief Returns the OpenCV type corresponding to this CudaMat.
 *
 * Uses cudaPixelTypeToCvType() to convert the stored CudaPixelType.
 *
 * @tparam T The CUDA pixel type.
 * @return OpenCV type constant.
 */
template <typename T>
constexpr int CudaMat<T>::type() const {
  return cudaPixelTypeToCvType(type_);
}

/**
 * @brief Returns the number of images in the batch.
 *
 * @tparam T The CUDA pixel type.
 * @return Batch size.
 */
template <typename T>
constexpr int CudaMat<T>::batch_size() const {
  return batch_size_;
}

template <typename T>
BaseScalar_t<T>* CudaMat<T>::data_raw() {
  return reinterpret_cast<BaseScalar_t<T>*>(d_data_);
}

template <typename T>
const BaseScalar_t<T>* CudaMat<T>::data_raw() const {
  return reinterpret_cast<const BaseScalar_t<T>*>(d_data_);
}

} // namespace hm
