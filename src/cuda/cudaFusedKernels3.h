#pragma once
#include "cupano/cuda/cudaStatus.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaMat.h"
#include <cuda_runtime.h>

namespace hm {
namespace pano {

class CanvasManager3;

namespace cuda {

/**
 * Launch fused kernel that remaps all three images directly to their full buffers
 * This eliminates the intermediate canvas write/read for soft seam blending
 */
template <typename T_pipeline, typename T_compute>
CudaStatus launchFusedRemapToFullKernel3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    CudaMat<T_compute>& cudaFull0,
    CudaMat<T_compute>& cudaFull1,
    CudaMat<T_compute>& cudaFull2,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream);

/**
 * Launch fused kernel that remaps all three images to canvas
 * Used for non-blended regions in soft seam mode
 */
template <typename T_pipeline>
CudaStatus launchFusedRemapToCanvasKernel3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    CudaMat<T_pipeline>& canvas,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream);

/**
 * Launch fused kernel for hard seam mode
 * Processes all three images in a single pass based on mask
 */
template <typename T_pipeline>
CudaStatus launchFusedRemapHardSeam3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    const CudaMat<unsigned char>& hardSeamMask,
    CudaMat<T_pipeline>& canvas,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream);

} // namespace cuda
} // namespace pano
} // namespace hm
