#pragma once
#include <cupano/gpu/gpu_runtime.h>
#include "cupano/cuda/cudaStatus.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaMat.h"

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
    int output_origin_x,
    int output_origin_y,
    const CanvasManager3& canvas_manager,
    cudaStream_t stream);

// Source-compatible full-canvas overload.
template <typename T_pipeline, typename T_compute>
inline CudaStatus launchFusedRemapToFullKernel3(
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
    cudaStream_t stream) {
  return launchFusedRemapToFullKernel3(
      inputImage0,
      inputImage1,
      inputImage2,
      remap_0_x,
      remap_0_y,
      remap_1_x,
      remap_1_y,
      remap_2_x,
      remap_2_y,
      cudaFull0,
      cudaFull1,
      cudaFull2,
      0,
      0,
      canvas_manager,
      stream);
}

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
    cudaStream_t stream);

} // namespace cuda
} // namespace pano
} // namespace hm
