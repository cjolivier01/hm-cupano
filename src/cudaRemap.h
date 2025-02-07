#pragma once

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
cudaError_t batched_remap_kernel(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in defR,
    T_in defG,
    T_in defB,
    int batchSize,
    cudaStream_t stream = 0);

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
    cudaStream_t stream = 0);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    cudaStream_t stream = 0);

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
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
    cudaStream_t stream = 0);
