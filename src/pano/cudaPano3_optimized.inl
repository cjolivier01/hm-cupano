#pragma once
#include <csignal>
#include "cupano/cuda/cudaFusedKernels3.h"
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano3.h"

namespace hm {
namespace pano {
namespace cuda {

/**
 * Optimized process_impl using fused kernels for THREE images
 * This implementation significantly reduces kernel launches and memory bandwidth
 */
template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process_impl_optimized(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    StitchingContext3<T_pipeline, T_compute>& stitch_context,
    const CanvasManager3& canvas_manager,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  assert(canvas);
  assert(inputImage0.batch_size() == stitch_context.batch_size());
  assert(inputImage1.batch_size() == stitch_context.batch_size());
  assert(inputImage2.batch_size() == stitch_context.batch_size());
  assert(canvas->batch_size() == stitch_context.batch_size());

  if (stitch_context.is_hard_seam() || stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream));
  }

  if (stitch_context.is_hard_seam()) {
    CUDA_RETURN_IF_ERROR(launchFusedRemapHardSeam3(
        inputImage0,
        inputImage1,
        inputImage2,
        *stitch_context.remap_0_x,
        *stitch_context.remap_0_y,
        *stitch_context.remap_1_x,
        *stitch_context.remap_1_y,
        *stitch_context.remap_2_x,
        *stitch_context.remap_2_y,
        *stitch_context.cudaBlendHardSeam,
        *canvas,
        canvas_manager,
        stream));
    return std::move(canvas);
  }

  if (stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(launchFusedRemapHardSeam3(
        inputImage0,
        inputImage1,
        inputImage2,
        *stitch_context.remap_0_x,
        *stitch_context.remap_0_y,
        *stitch_context.remap_1_x,
        *stitch_context.remap_1_y,
        *stitch_context.remap_2_x,
        *stitch_context.remap_2_y,
        *stitch_context.cudaBlendHardSeam,
        *canvas,
        canvas_manager,
        stream));
  }

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull0->data(), 0, stitch_context.cudaFull0->size(), stream));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull1->data(), 0, stitch_context.cudaFull1->size(), stream));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull2->data(), 0, stitch_context.cudaFull2->size(), stream));
  CUDA_RETURN_IF_ERROR(launchFusedRemapToFullKernel3(
      inputImage0,
      inputImage1,
      inputImage2,
      *stitch_context.remap_0_x,
      *stitch_context.remap_0_y,
      *stitch_context.remap_1_x,
      *stitch_context.remap_1_y,
      *stitch_context.remap_2_x,
      *stitch_context.remap_2_y,
      *stitch_context.cudaFull0,
      *stitch_context.cudaFull1,
      *stitch_context.cudaFull2,
      stitch_context.minimizes_blend ? stitch_context.blend_roi_canvas.x : 0,
      stitch_context.minimizes_blend ? stitch_context.blend_roi_canvas.y : 0,
      canvas_manager,
      stream));

  CudaMat<T_compute>& blended = *stitch_context.cudaFull0;
  CUDA_RETURN_IF_ERROR(cudaBatchedLaplacianBlendWithContext3(
      stitch_context.cudaFull0->data_raw(),
      stitch_context.cudaFull1->data_raw(),
      stitch_context.cudaFull2->data_raw(),
      stitch_context.cudaBlendSoftSeam->data_raw(),
      blended.data_raw(),
      *stitch_context.laplacian_blend_context,
      stitch_context.cudaFull0->channels(),
      stream));
  const cv::Rect write_roi = stitch_context.minimizes_blend ? stitch_context.write_roi_canvas
                                                            : cv::Rect(0, 0, blended.width(), blended.height());
  const int src_x = stitch_context.minimizes_blend ? write_roi.x - stitch_context.blend_roi_canvas.x : 0;
  const int src_y = stitch_context.minimizes_blend ? write_roi.y - stitch_context.blend_roi_canvas.y : 0;
  const CudaStatus copy_status = copy_roi_batched<T_compute, T_pipeline>(
      blended.surface(),
      write_roi.width,
      write_roi.height,
      src_x,
      src_y,
      canvas->surface(),
      write_roi.x,
      write_roi.y,
      stitch_context.batch_size(),
      stream);
  CUDA_RETURN_IF_ERROR(copy_status);
  return std::move(canvas);
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::
    process_impl_optimized_current(
        const CudaMat<T_pipeline>& inputImage0,
        const CudaMat<T_pipeline>& inputImage1,
        const CudaMat<T_pipeline>& inputImage2,
        cudaStream_t stream,
        std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  return process_impl_optimized(
      inputImage0, inputImage1, inputImage2, *stitch_context_, *canvas_manager_, stream, std::move(canvas));
}

/**
 * Updated process method to use optimized implementation
 */
template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process_optimized(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CUDA_RETURN_IF_ERROR(status_);

  // Use optimized implementation by default
  auto result = process_impl_optimized_current(inputImage0, inputImage1, inputImage2, stream, std::move(canvas));

  if (!result.ok()) {
    status_.Update(result.status());
  }

  return result;
}

} // namespace cuda
} // namespace pano
} // namespace hm
