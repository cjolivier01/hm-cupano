#include <cuda_runtime.h>
#include <cassert>

// If you want to support 16‐bit and bfloat16 types:
#include <cuda_bf16.h>
#include <cuda_fp16.h>

////////////////////////////////////////////////////////////////////////////////
// Templated Device Kernels
////////////////////////////////////////////////////////////////////////////////

#ifndef HALF3_DEFINED
#define HALF3_DEFINED
/**
 * @brief 3-element vector of __half values.
 */
struct half3 {
  __half x, y, z;
};
#endif

namespace {

template <typename F>
__device__ inline unsigned char round_to_uchar(const F& x) {
  F x_rounded = x + F(0.5); // Add 0.5 in the type's precision
  if (x_rounded <= F(0.0)) {
    return 0;
  }
  if (x_rounded >= F(255.0)) {
    return 255;
  }
  return static_cast<unsigned char>(x_rounded); // Cast result to unsigned char
}

template <typename T_dest, typename T_src>
__device__ inline T_dest perform_cast(const T_src& src) {
  return static_cast<T_dest>(src);
}

template <>
__device__ inline uchar3 perform_cast(const float3& src) {
  return uchar3{
      .x = static_cast<unsigned char>(round_to_uchar(src.x)),
      .y = static_cast<unsigned char>(round_to_uchar(src.y)),
      .z = static_cast<unsigned char>(round_to_uchar(src.z)),
  };
}

template <>
__device__ inline uchar3 perform_cast(const half3& src) {
  return uchar3{
      .x = static_cast<unsigned char>(round_to_uchar(src.x)),
      .y = static_cast<unsigned char>(round_to_uchar(src.y)),
      .z = static_cast<unsigned char>(round_to_uchar(src.z)),
  };
}

template <>
__device__ inline half3 perform_cast(const uchar3& src) {
  return half3{
      .x = static_cast<unsigned char>(round_to_uchar(src.x)),
      .y = static_cast<unsigned char>(round_to_uchar(src.y)),
      .z = static_cast<unsigned char>(round_to_uchar(src.z)),
  };
}
} // namespace
/**
 * @brief Templated batched kernel to fill an image (or batch of images) with a constant value.
 *
 * Each image in the batch is assumed to be stored in row–major order.
 *
 * @tparam T Numeric type (e.g. float, __half, __nv_bfloat16, unsigned char, etc.)
 * @param dest Pointer to destination images in device memory.
 * @param destWidth Width of each destination image.
 * @param destHeight Height of each destination image.
 * @param channels Number of channels per pixel.
 * @param value Constant value to fill.
 * @param batchSize Number of images in the batch.
 */
template <typename T>
__global__ void fillKernelBatched(T* dest, int destWidth, int destHeight, int channels, T value, int batchSize) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int b = blockIdx.z;
  if (b < batchSize && x < destWidth && y < destHeight) {
    int offset = b * (destWidth * destHeight * channels);
    int idx = (y * destWidth + x) * channels;
    for (int c = 0; c < channels; ++c) {
      dest[offset + idx + c] = value;
    }
  }
}

/**
 * @brief Templated batched kernel to copy a region of interest (ROI) from a source image
 *        to a destination canvas while performing a type conversion.
 *
 * For each image in the batch, the kernel copies a rectangular region defined by a source ROI
 * into the destination image. The source pixel values (of type T_in) are converted to type T_out.
 *
 * @tparam T_in  Input pixel type.
 * @tparam T_out Output pixel type.
 * @param src Pointer to the batch of source images in device memory.
 * @param full_src_width Full width of each source image.
 * @param full_src_height Full height of each source image.
 * @param regionWidth Width of the ROI to copy.
 * @param regionHeight Height of the ROI to copy.
 * @param srcROI_x X-coordinate of the top-left corner of the ROI in the source images.
 * @param srcROI_y Y-coordinate of the top-left corner of the ROI in the source images.
 * @param dest Pointer to the batch of destination images in device memory.
 * @param destWidth Width of each destination image.
 * @param destHeight Height of each destination image.
 * @param offsetX X-coordinate in the destination image where the ROI is pasted.
 * @param offsetY Y-coordinate in the destination image where the ROI is pasted.
 * @param channels Number of channels per pixel.
 * @param batchSize Number of images in the batch.
 */
template <typename T_in, typename T_out>
__global__ void copyRoiKernelBatched(
    const T_in* src,
    int full_src_width,
    int full_src_height,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    T_out* dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < regionWidth && y < regionHeight) {
    int srcX = srcROI_x + x;
    int srcY = srcROI_y + y;
    if (srcX < full_src_width && srcY < full_src_height) {
      int srcOffset = b * (full_src_width * full_src_height * channels);
      int srcIdx = (srcY * full_src_width + srcX) * channels;
      int destX = offsetX + x;
      int destY = offsetY + y;
      if (destX < destWidth && destY < destHeight) {
        int destOffset = b * (destWidth * destHeight * channels);
        int destIdx = (destY * destWidth + destX) * channels;
        for (int c = 0; c < channels; ++c) {
          dest[destOffset + destIdx + c] = perform_cast<T_out>(src[srcOffset + srcIdx + c]);
          // dest[destOffset + destIdx + c] = static_cast<T_out>(src[srcOffset + srcIdx + c]);
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Templated Host Functions
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief Creates full canvas images by copying specified source ROIs from a batch of images (and masks)
 *        into preallocated destination canvases.
 *
 * This function fills the destination canvases with default values (for images, 0; for masks, 1)
 * and then copies the ROI from each source image (and source mask) into the corresponding destination canvas.
 *
 * @tparam T Numeric type for images (e.g., float, __half, __nv_bfloat16).
 * @tparam U Numeric type for masks (typically unsigned char).
 * @param d_imgs Pointer to the batch of source images in device memory.
 * @param src_full_width Full width of each source image.
 * @param src_full_height Full height of each source image.
 * @param region_width Width of the ROI to copy from each source image.
 * @param region_height Height of the ROI to copy from each source image.
 * @param channels Number of channels in the source images.
 * @param d_masks Pointer to the batch of source masks in device memory (or nullptr if not provided).
 * @param mask_width Width of each source mask.
 * @param mask_height Height of each source mask.
 * @param mask_channels Number of channels in the source masks.
 * @param src_roi_x X-coordinate of the top-left corner of the ROI in the source images/masks.
 * @param src_roi_y Y-coordinate of the top-left corner of the ROI in the source images/masks.
 * @param x Reference to destination X-offset for the ROI in the destination canvases (may be adjusted).
 * @param y Reference to destination Y-offset for the ROI in the destination canvases (may be adjusted).
 * @param canvas_w Width of the destination canvases.
 * @param canvas_h Height of the destination canvases.
 * @param batchSize Number of images (and masks) in the batch.
 * @param d_full_imgs Preallocated pointer to the destination canvases for images in device memory.
 * @param d_full_masks Preallocated pointer to the destination canvases for masks in device memory (or nullptr).
 * @param stream CUDA stream to use for kernel launches.
 * @return cudaError_t The CUDA error code after kernel launches.
 */
template <typename T_in, typename T_out, typename U>
cudaError_t simple_make_full_batch(
    const T_in* d_imgs,
    int src_full_width,
    int src_full_height,
    int region_width,
    int region_height,
    int channels,
    const U* d_masks,
    int mask_width,
    int mask_height,
    int mask_channels,
    int src_roi_x,
    int src_roi_y,
    int destOffsetX,
    int destOffsetY,
    int canvas_w,
    int canvas_h,
    bool adjust_origin,
    int batchSize,
    T_out* d_full_imgs,
    U* d_full_masks,
    cudaStream_t stream) {
  // Ensure the destination offsets are nonnegative.
  assert(destOffsetX >= 0 && destOffsetX >= 0);

  // Define kernel launch parameters.
  dim3 blockDim(16, 16, 1);
  dim3 gridDimCanvas((canvas_w + blockDim.x - 1) / blockDim.x, (canvas_h + blockDim.y - 1) / blockDim.y, batchSize);

  // -------------------------------------------------------
  // Fill the destination canvases with default values.
  // -------------------------------------------------------
  // For images: fill with 0.
  fillKernelBatched<T_out><<<gridDimCanvas, blockDim, 0, stream>>>(
      d_full_imgs, canvas_w, canvas_h, channels, static_cast<T_out>(0), batchSize);
  // For masks (if provided): fill with 1.
  if (d_masks && d_full_masks) {
    fillKernelBatched<U><<<gridDimCanvas, blockDim, 0, stream>>>(
        d_full_masks, canvas_w, canvas_h, mask_channels, static_cast<U>(1), batchSize);
  }

  // -------------------------------------------------------
  // Copy the ROI from each source image/mask into the destination canvases.
  // -------------------------------------------------------
  dim3 gridDimCopy(
      (region_width + blockDim.x - 1) / blockDim.x, (region_height + blockDim.y - 1) / blockDim.y, batchSize);

  // Copy the ROI for the images.
  // Here we use the same type for input and output.
  copyRoiKernelBatched<T_in, T_out><<<gridDimCopy, blockDim, 0, stream>>>(
      d_imgs,
      src_full_width,
      src_full_height,
      region_width,
      region_height,
      src_roi_x,
      src_roi_y,
      d_full_imgs,
      canvas_w,
      canvas_h,
      destOffsetX,
      destOffsetY,
      channels,
      batchSize);

  // Copy the ROI for the masks (if provided).
  if (d_masks && d_full_masks) {
    copyRoiKernelBatched<U, U><<<gridDimCopy, blockDim, 0, stream>>>(
        d_masks,
        mask_width,
        mask_height,
        region_width,
        region_height,
        src_roi_x,
        src_roi_y,
        d_full_masks,
        canvas_w,
        canvas_h,
        destOffsetX,
        destOffsetY,
        mask_channels,
        batchSize);
  }
  return cudaGetLastError();
}

/**
 * @brief Interface function for launching the batched ROI copy kernel for images,
 *        with separate input and output types.
 *
 * This function sets up grid and block dimensions and launches the
 * copyRoiKernelBatched kernel.
 *
 * @tparam T_in  Input pixel type.
 * @tparam T_out Output pixel type.
 * @param d_src Pointer to the batch of source images in device memory.
 * @param full_src_width Full width of each source image.
 * @param full_src_height Full height of each source image.
 * @param regionWidth Width of the ROI to copy.
 * @param regionHeight Height of the ROI to copy.
 * @param srcROI_x X-coordinate of the top-left corner of the ROI in the source images.
 * @param srcROI_y Y-coordinate of the top-left corner of the ROI in the source images.
 * @param d_dest Pointer to the batch of destination images in device memory.
 * @param destWidth Width of each destination image.
 * @param destHeight Height of each destination image.
 * @param offsetX X-coordinate in the destination image where the ROI is pasted.
 * @param offsetY Y-coordinate in the destination image where the ROI is pasted.
 * @param channels Number of channels per pixel.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for the kernel launch.
 * @return cudaError_t The CUDA error code after kernel launch.
 */
template <typename T_in, typename T_out>
cudaError_t copyRoiBatchedInterface(
    const T_in* d_src,
    int full_src_width,
    int full_src_height,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    T_out* d_dest,
    int destWidth,
    int destHeight,
    int offsetX,
    int offsetY,
    int channels,
    int batchSize,
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((regionWidth + blockDim.x - 1) / blockDim.x, (regionHeight + blockDim.y - 1) / blockDim.y, batchSize);
  copyRoiKernelBatched<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      d_src,
      full_src_width,
      full_src_height,
      regionWidth,
      regionHeight,
      srcROI_x,
      srcROI_y,
      d_dest,
      destWidth,
      destHeight,
      offsetX,
      offsetY,
      channels,
      batchSize);
  return cudaGetLastError();
}

////////////////////////////////////////////////////////////////////////////////
// Explicit Template Instantiations
////////////////////////////////////////////////////////////////////////////////

#define INSTANTIATE_FILL_KERNEL_BATCHED(T)       \
  template __global__ void fillKernelBatched<T>( \
      T * dest, int destWidth, int destHeight, int channels, T value, int batchSize);

#define INSTANTIATE_COPY_ROI_KERNEL_BATCHED(Tin, Tout)      \
  template __global__ void copyRoiKernelBatched<Tin, Tout>( \
      const Tin* src,                                       \
      int full_src_width,                                   \
      int full_src_height,                                  \
      int regionWidth,                                      \
      int regionHeight,                                     \
      int srcROI_x,                                         \
      int srcROI_y,                                         \
      Tout* dest,                                           \
      int destWidth,                                        \
      int destHeight,                                       \
      int offsetX,                                          \
      int offsetY,                                          \
      int channels,                                         \
      int batchSize);

#define INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(Tin, Tout, U)     \
  template cudaError_t simple_make_full_batch<Tin, Tout, U>( \
      const Tin* d_imgs,                                     \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      const U* d_masks,                                      \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      int,                                                   \
      bool,                                                  \
      int,                                                   \
      Tout* d_full_imgs,                                     \
      U* d_full_masks,                                       \
      cudaStream_t stream);

#define INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(Tin, Tout)  \
  template cudaError_t copyRoiBatchedInterface<Tin, Tout>( \
      const Tin* d_src,                                    \
      int full_src_width,                                  \
      int full_src_height,                                 \
      int regionWidth,                                     \
      int regionHeight,                                    \
      int srcROI_x,                                        \
      int srcROI_y,                                        \
      Tout* d_dest,                                        \
      int destWidth,                                       \
      int destHeight,                                      \
      int offsetX,                                         \
      int offsetY,                                         \
      int channels,                                        \
      int batchSize,                                       \
      cudaStream_t stream);

// --- Device kernels ---

// fillKernelBatched instantiations:
INSTANTIATE_FILL_KERNEL_BATCHED(float)
INSTANTIATE_FILL_KERNEL_BATCHED(float3)
INSTANTIATE_FILL_KERNEL_BATCHED(__half)
INSTANTIATE_FILL_KERNEL_BATCHED(__nv_bfloat16)

// copyRoiKernelBatched instantiations:

// Same–type instantiations:
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(float, float)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(float3, float3)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(uchar3, uchar3)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(uchar3, half3)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(__half, __half)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(__nv_bfloat16, __nv_bfloat16)

// Conversion instantiations:
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(float, __half)
INSTANTIATE_COPY_ROI_KERNEL_BATCHED(__half, float)

// --- Host functions ---

// simple_make_full_batch instantiations:
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(unsigned char, float, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(unsigned char, __half, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(unsigned char, __nv_bfloat16, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(float, float, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(unsigned char, unsigned char, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(__half, __half, unsigned char)
INSTANTIATE_SIMPLE_MAKE_FULL_BATCH(__nv_bfloat16, __nv_bfloat16, unsigned char)

// copyRoiBatchedInterface instantiations:

// Same–type instantiations:
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(unsigned char, float)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(unsigned char, __half)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(half3, uchar3)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(unsigned char, __nv_bfloat16)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(float, float)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(float3, float3)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(float3, uchar3)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(uchar3, uchar3)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(__half, __half)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(__nv_bfloat16, __nv_bfloat16)

// Conversion instantiations:
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(float, __half)
INSTANTIATE_COPY_ROI_BATCHED_INTERFACE(__half, float)
