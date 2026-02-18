#pragma once

#include "src/cuda/cudaTypes.h"

#include <cuda_runtime.h>

/**
 * @brief Batched remap host function.
 *
 * Launches the BatchedRemapKernel to process a batch of images using mapping arrays.
 *
 * @param d_src Device pointer to the batch of source images.
 * @param srcW Width of each source image.
 * @param srcH Height of each source image.
 * @param d_dest Device pointer to the batch of destination images.
 * @param destW Width of each destination image.
 * @param destH Height of each destination image.
 * @param d_mapX Device pointer to the batch of mapping arrays for X coordinates.
 * @param d_mapY Device pointer to the batch of mapping arrays for Y coordinates.
 * @param defR Default red component.
 * @param defG Default green component.
 * @param defB Default blue component.
 * @param batchSize Number of images in the batch.
 * @param stream CUDA stream to use for the kernel launch (default is 0).
 * @return cudaError_t The status returned by cudaGetLastError.
 */
template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in dflt,
    int batchSize,
    cudaStream_t stream);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write,
    cudaStream_t stream);

/**
 * @brief Batched remap for a rectangular ROI inside the remap map.
 *
 * This is equivalent to `batched_remap_kernel_ex_offset()`, but only processes a
 * sub-rectangle `[roiX, roiY, roiW, roiH]` in the remap map (and thus only writes
 * the corresponding region in `dest`).
 *
 * ROI coordinates are expressed in the remap-map coordinate system (i.e. local
 * to the remapped image). The `offsetX/offsetY` apply to the full remap map, and
 * destination coordinates are computed as:
 *   `destX = offsetX + (roiX + x)`
 *   `destY = offsetY + (roiY + y)`
 *
 * @tparam T_in  Input pixel type (source surface).
 * @tparam T_out Output pixel type (destination surface).
 */
template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_roi(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    bool no_unmapped_write,
    cudaStream_t stream);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_adjust(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write,
    float3 adjustment,
    cudaStream_t stream);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    cudaStream_t stream);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    float3 adjustment,
    cudaStream_t stream);

/**
 * @brief Fused hard-seam remap for N images.
 *
 * For each output pixel, selects the contributing image from `dest_image_map`, then applies that image's
 * remap to sample from the corresponding input surface. Pixels with unmapped/invalid coordinates are
 * left unchanged, so callers typically `cudaMemsetAsync()` the destination to 0 before invoking.
 *
 * All pointer arrays (`d_inputs`, `d_mapX_ptrs`, `d_mapY_ptrs`, `d_offsets`, `d_sizes`) must live in
 * device memory and have length `n_images`.
 */
template <typename T>
cudaError_t batched_remap_hard_seam_kernel_n(
    const CudaSurface<T>* d_inputs,
    const unsigned short* const* d_mapX_ptrs,
    const unsigned short* const* d_mapY_ptrs,
    const int2* d_offsets,
    const int2* d_sizes,
    int n_images,
    const unsigned char* dest_image_map,
    CudaSurface<T> dest,
    int batchSize,
    cudaStream_t stream);
