#pragma once

#include <cuda_runtime.h>

#include "cudaTypes.h"

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
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for the kernel launch.
 * @return cudaError_t The CUDA error code after kernel launch.
 */
template <typename T_in, typename T_out>
cudaError_t copy_roi_batched(
    const CudaSurface<T_in>& src,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    CudaSurface<T_out> dest,
    int offsetX,
    int offsetY,
    int batchSize,
    cudaStream_t stream);

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
 * @param d_masks Pointer to the batch of source masks in device memory (or nullptr if not provided).
 * @param mask_width Width of each source mask.
 * @param mask_height Height of each source mask.
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
template <typename T_in, typename T_out>
cudaError_t simple_make_full_batch(
    const CudaSurface<T_in>& src,
    int region_width,
    int region_height,
    int src_roi_x,
    int src_roi_y,
    int destOffsetX,
    int destOffsetY,
    bool adjust_origin,
    int batchSize,
    CudaSurface<T_out> dest,
    cudaStream_t stream);

/**
 * @brief Launch the AlphaConditionalCopyKernel for surfaces that use a vector type with an alpha channel.
 *
 * @tparam T CUDA vector type (e.g. uchar4, float4, half4).
 * @param image1 Destination/source surface for the first image.
 * @param image2 Destination/source surface for the second image.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use.
 * @return cudaError_t cudaGetLastError() after launching the kernel.
 */
template <typename T>
cudaError_t AlphaConditionalCopy(CudaSurface<T>& image1, CudaSurface<T>& image2, int batchSize, cudaStream_t stream);
