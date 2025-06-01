#pragma once

#include "cudaMat.h"

#include <cuda_runtime.h>
#include <cassert>

namespace hm {

// Helper function: Returns the number of channels expected for a given CUDA pixel type.
// This function maps each CUDA pixel type enumeration (e.g., single-channel, 3-channel, 4-channel)
// to the corresponding number of channels.
inline int cudaPixelTypeChannels(CudaPixelType fmt) {
  switch (fmt) {
    // Unsigned char pixel types.
    case CUDA_PIXEL_UCHAR1:
      return 1; // 1 channel (e.g., grayscale)
    case CUDA_PIXEL_UCHAR3:
      return 3; // 3 channels (e.g., RGB)
    case CUDA_PIXEL_UCHAR4:
      return 4; // 4 channels (e.g., RGBA)
    // Unsigned short pixel types.
    case CUDA_PIXEL_USHORT1:
      return 1;
    case CUDA_PIXEL_USHORT3:
      return 3;
    case CUDA_PIXEL_USHORT4:
      return 4;
    // Integer pixel types.
    case CUDA_PIXEL_INT1:
      return 1;
    case CUDA_PIXEL_INT3:
      return 3;
    case CUDA_PIXEL_INT4:
      return 4;
    // Floating point pixel types.
    case CUDA_PIXEL_FLOAT1:
      return 1;
    case CUDA_PIXEL_FLOAT3:
      return 3;
    case CUDA_PIXEL_FLOAT4:
      return 4;
    // Half precision floating point pixel types.
    case CUDA_PIXEL_HALF1:
      return 1;
    case CUDA_PIXEL_HALF3:
      return 3;
    case CUDA_PIXEL_HALF4:
      return 4;
    // BF16 (Brain Floating Point) pixel types.
    case CUDA_PIXEL_BF16_1:
      return 1;
    case CUDA_PIXEL_BF16_3:
      return 3;
    case CUDA_PIXEL_BF16_4:
      return 4;
    // If the pixel type is unrecognized, return 0.
    default:
      return 0;
  }
}

/**
 * @brief Constructs a CudaMat from a single cv::Mat.
 *
 * This constructor takes an OpenCV Mat, allocates device (GPU) memory for it, and optionally
 * copies the data from the host (CPU) to the device. It also validates that the input Mat's
 * element size matches the expected CUDA pixel type.
 *
 * @tparam T The CUDA pixel type.
 * @param mat The input cv::Mat.
 * @param copy If true, copies the data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const cv::Mat& mat, bool copy) : rows_(mat.rows), cols_(mat.cols), batch_size_(1) {
  // Convert the cv::Mat type (an integer) to the corresponding CudaPixelType.
  type_ = cvMatToCudaPixelType(mat);
  // Get the expected element size (in bytes) for the specified CUDA pixel type.
  size_t expectedElemSize = cudaPixelElementSize(type_);
  // Ensure the cv::Mat element size matches what is expected for this pixel type.
  assert(mat.elemSize() == expectedElemSize);
  // Calculate the total amount of memory required.
  size_t total_size = mat.total() * expectedElemSize;
  // Allocate memory on the device (GPU).
  cudaMalloc(&d_data_, total_size);
  // Verify that the cv::Mat data is stored in continuous memory.
  assert(mat.isContinuous());
  // If requested, copy the image data from host to device.
  if (copy) {
    cudaMemcpy(d_data_, mat.data, total_size, cudaMemcpyHostToDevice);
  }
}

/**
 * @brief Constructs a CudaMat from a batch of cv::Mat images.
 *
 * This constructor handles multiple images (batch processing). It allocates one contiguous block
 * of device memory to store all images and, if requested, copies each image from the host to the device.
 *
 * @tparam T The CUDA pixel type.
 * @param mat_batch Vector of input cv::Mat images.
 * @param copy If true, copies data to device memory.
 */
template <typename T>
CudaMat<T>::CudaMat(const std::vector<cv::Mat>& mat_batch, bool copy)
    : batch_size_(static_cast<int>(mat_batch.size())) {
  // Ensure that there is at least one image in the batch.
  assert(batch_size_ > 0);
  // Use the first image to determine dimensions and pixel type.
  const cv::Mat& first = mat_batch.at(0);
  rows_ = first.rows;
  cols_ = first.cols;
  // Convert the cv::Mat type to the CUDA pixel type.
  type_ = cvMatToCudaPixelType(first);
  // Calculate the expected size of each element in the image.
  size_t expectedElemSize = cudaPixelElementSize(type_);
  // Verify that the first image's element size matches the expected size.
  assert(first.elemSize() == expectedElemSize);
  // Calculate the memory size required for one image.
  size_t size_each = first.total() * expectedElemSize;
  // Calculate the total memory size required for all images.
  size_t total_size = size_each * batch_size_;
  // Allocate device memory for the entire batch.
  cudaError_t cuerr = cudaMalloc(&d_data_, total_size);
  // If allocation succeeds and data copy is requested, copy each image.
  if (cuerr == cudaError_t::cudaSuccess && copy) {
    uint8_t* p = reinterpret_cast<uint8_t*>(d_data_);
    for (const cv::Mat& mat : mat_batch) {
      // Ensure the image is in continuous memory.
      assert(mat.isContinuous());
      // Verify that each image in the batch has the same dimensions.
      assert(mat.rows == rows_ && mat.cols == cols_);
      // Confirm that the image's element size matches the expected size.
      assert(mat.elemSize() == expectedElemSize);
      // Copy the image data from host to the appropriate location in device memory.
      cudaMemcpy(p, mat.data, size_each, cudaMemcpyHostToDevice);
      // Move the pointer to the next image location.
      p += size_each;
    }
  }
}

/**
 * @brief Constructs a CudaMat with explicit dimensions and pixel type.
 *
 * This constructor is used when the image dimensions (batch size, width, height) and the pixel type
 * are explicitly provided. It allocates device memory for the batch and checks that the channel count
 * provided matches the expected channel count for the pixel type.
 *
 * @tparam T The CUDA pixel type.
 * @param B Batch size (number of images).
 * @param W Image width.
 * @param H Image height.
 * @param C Number of channels.
 * @param type The CUDA pixel type.
 */
template <typename T>
CudaMat<T>::CudaMat(int B, int W, int H, int C, CudaPixelType type) : batch_size_(B), rows_(H), cols_(W), type_(type) {
  // Determine the expected number of channels for the given CUDA pixel type.
  int expectedChannels = cudaPixelTypeChannels(type_);
  // Ensure the provided channel count is correct.
  assert(expectedChannels == C);
  // Get the size (in bytes) of one element.
  size_t elemSize = cudaPixelElementSize(type_);
  // Check that the size of the template type T matches the expected element size.
  assert(sizeof(T) == elemSize);
  // Compute the total memory size needed.
  const size_t total_size = static_cast<size_t>(B * W * H) * elemSize;
  // Allocate memory on the device.
  cudaMalloc(&d_data_, total_size);
}

/**
 * @brief Constructs a CudaMat with explicit dimensions, with the pixel type automatically inferred from template T.
 *
 * This constructor infers the pixel type from T and validates that the provided channel count is consistent
 * with the inferred type.
 *
 * @tparam T The CUDA pixel type.
 * @param B Batch size.
 * @param W Image width.
 * @param H Image height.
 * @param C Number of channels (used for validation).
 */
template <typename T>
CudaMat<T>::CudaMat(int B, int W, int H, int C)
    : rows_(H),
      cols_(W),
      type_(CudaTypeToPixelType<T>::value), // Inferred from T
      batch_size_(B) {
  // Validate that the number of channels matches expectation,
  // taking into account the size differences between T and its base scalar type.
  assert(static_cast<size_t>(cudaPixelTypeChannels(type_)) == C * sizeof(T) / sizeof(typename BaseScalar<T>::type));
  size_t elemSize = cudaPixelElementSize(type_);
  // Ensure that T has the correct size.
  assert(sizeof(T) == elemSize);
  // Allocate device memory based on the total number of pixels and element size.
  size_t total_size = static_cast<size_t>(B * W * H) * elemSize;
  cudaMalloc(&d_data_, total_size);
}

/**
 * @brief Constructs a CudaMat using an already allocated device pointer.
 *
 * This constructor allows wrapping an existing device pointer into a CudaMat without allocating new memory.
 * The memory is not owned by this instance, so it will not be freed upon destruction.
 *
 * @tparam T The CUDA pixel type.
 * @param d_data Pre-allocated device pointer.
 * @param B Batch size.
 * @param W Image width.
 * @param H Image height.
 * @param C Number of channels (used for validation).
 */
template <typename T>
CudaMat<T>::CudaMat(T* d_data, int B, int W, int H, int C)
    : d_data_(d_data),
      rows_(H),
      cols_(W),
      type_(CudaTypeToPixelType<T>::value), // Pixel type inferred from T
      batch_size_(B),
      owns_(false) { // This instance does not own the memory.
  // Verify that the number of channels matches the expected value.
  assert(static_cast<size_t>(cudaPixelTypeChannels(type_)) == C * sizeof(T) / sizeof(typename BaseScalar<T>::type));
  // Confirm that the size of T is as expected.
  assert(sizeof(T) == cudaPixelElementSize(type_));
}

/**
 * @brief Constructs a CudaMat from a SurfaceInfo structure.
 *
 * This constructor creates a CudaMat based on externally provided surface information,
 * which includes a device pointer, dimensions, and optional pitch (stride).
 * The CudaMat instance does not own the memory.
 *
 * @tparam T The CUDA pixel type.
 * @param surface_info Structure containing device pointer, width, height, and optional pitch.
 * @param B Batch size.
 */
template <typename T>
CudaMat<T>::CudaMat(const SurfaceInfo& surface_info, int B) : type_(CudaTypeToPixelType<T>::value) {
  // Validate that the surface information contains valid pointers and dimensions.
  assert(surface_info.data_ptr && surface_info.width && surface_info.height && B);
  // Check that if a pitch is provided, it is sufficient for the given width.
  assert(!surface_info.pitch || surface_info.pitch >= (int)sizeof(T) * surface_info.width);
  // Set dimensions from the surface info.
  rows_ = surface_info.height;
  cols_ = surface_info.width;
  pitch_ = surface_info.pitch;
  // Assign the existing device pointer.
  d_data_ = static_cast<T*>(surface_info.data_ptr);
  batch_size_ = B;
  // Indicate that this instance does not own the memory.
  owns_ = false;
}

/**
 * @brief Destructor for CudaMat.
 *
 * Frees the allocated device memory if this instance owns it.
 *
 * @tparam T The CUDA pixel type.
 */
template <typename T>
CudaMat<T>::~CudaMat() {
  // Free the device memory only if it was allocated by this instance.
  if (d_data_ && owns_) {
    cudaFree(d_data_);
  }
}

/**
 * @brief Downloads an image from device memory to a cv::Mat.
 *
 * Downloads the image corresponding to the specified batch index from the GPU back to the CPU.
 * This method converts the stored CUDA pixel format to an OpenCV-compatible type.
 *
 * @tparam T The CUDA pixel type.
 * @param batch_item The index of the image in the batch (default is 0).
 * @return A cv::Mat containing the downloaded image.
 */
template <typename T>
cv::Mat CudaMat<T>::download(int batch_item) const {
  // Validate that the requested batch index is within range.
  assert(batch_item >= 0 && batch_item < batch_size_);
  // Convert the stored CUDA pixel type to an OpenCV type constant.
  int cvType = cudaPixelTypeToCvType(type_);
  // Ensure the pitch is aligned to the size of T.
  assert(pitch() % sizeof(T) == 0);
  // Calculate the number of elements per row based on the pitch.
  int pitch_cols = pitch() / sizeof(T);
  // Verify that the pitch is sufficient for the expected number of columns.
  assert(pitch_cols >= cols_);

  // Create an OpenCV Mat with the dimensions and type that includes the pitch.
  cv::Mat mat(rows_, pitch_cols, cvType);
  // Calculate the size in bytes for one element and the total size for one image.
  size_t elemSize = cudaPixelElementSize(type_);
  size_t size_each = static_cast<size_t>(rows_ * cols_) * elemSize;
  // Calculate the starting address for the desired batch item.
  const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(d_data_) + batch_item * size_each;
  // Copy the data from device memory (GPU) to the host memory (CPU).
  cudaMemcpy(mat.data, src_ptr, size_each, cudaMemcpyDeviceToHost);
  // If the pitch is greater than the number of columns, extract the valid region.
  if (pitch_cols != cols_) {
    mat = mat(cv::Rect(0, 0, cols_, rows_));
  }
  return mat;
}

template <typename T>
cudaError_t CudaMat<T>::upload(const cv::Mat& cpu_mat, int batch_item, cudaStream_t stream) {
  // Validate that the requested batch index is within range.
  assert(batch_item >= 0 && batch_item < batch_size_);
  // Convert the stored CUDA pixel type to an OpenCV type constant.
  int cvType = cudaPixelTypeToCvType(type_);
  // Ensure the pitch is aligned to the size of T.
  assert(pitch() % sizeof(T) == 0);
  // Calculate the number of elements per row based on the pitch.
  int pitch_cols = pitch() / sizeof(T);
  // Verify that the pitch is sufficient for the expected number of columns.
  
  assert(pitch_cols == cols_);  // not supporting differeing pitch atm

  size_t elemSize = cudaPixelElementSize(type_);
  size_t size_each = static_cast<size_t>(rows_ * cols_) * elemSize;
  // Calculate the starting address for the desired batch item.
  assert(d_data_);
  uint8_t* src_ptr = reinterpret_cast<uint8_t*>(d_data_) + batch_item * size_each;
  // Copy the data from device memory (GPU) to the host memory (CPU).
  if (!stream) {
    return cudaMemcpy(src_ptr, cpu_mat.data, size_each, cudaMemcpyHostToDevice);
  } else {
    return cudaMemcpyAsync(src_ptr, cpu_mat.data, size_each, cudaMemcpyHostToDevice, stream);
  }
}

/**
 * @brief Returns a pointer to the device memory.
 *
 * This method provides direct access to the GPU memory.
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
 * This method provides read-only access to the GPU memory.
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
 * Converts the stored CUDA pixel type to an OpenCV constant using cudaPixelTypeToCvType().
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

/**
 * @brief Returns a pointer to the raw data as the base scalar type.
 *
 * Useful for operations that require access to the underlying data type representation.
 *
 * @tparam T The CUDA pixel type.
 * @return Pointer to raw data.
 */
template <typename T>
BaseScalar_t<T>* CudaMat<T>::data_raw() {
  return reinterpret_cast<BaseScalar_t<T>*>(d_data_);
}

/**
 * @brief Returns a const pointer to the raw data as the base scalar type.
 *
 * @tparam T The CUDA pixel type.
 * @return Const pointer to raw data.
 */
template <typename T>
const BaseScalar_t<T>* CudaMat<T>::data_raw() const {
  return reinterpret_cast<const BaseScalar_t<T>*>(d_data_);
}

} // namespace hm
