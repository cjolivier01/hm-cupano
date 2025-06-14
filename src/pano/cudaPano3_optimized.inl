#pragma once
#include <csignal>
#include <optional>
#include "cupano/cuda/cudaFusedKernels3.h"
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/utils/cudaBlendShow.h"
#include "cupano/utils/showImage.h"

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
    const std::optional<float3>& image_adjustment,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CudaStatus cuerr;
  assert(canvas);
  assert(inputImage0.batch_size() == stitch_context.batch_size());
  assert(inputImage1.batch_size() == stitch_context.batch_size());
  assert(inputImage2.batch_size() == stitch_context.batch_size());
  assert(canvas->batch_size() == stitch_context.batch_size());

  // Clear canvas for RGBA formats
  if constexpr (sizeof(T_pipeline) / sizeof(BaseScalar_t<T_pipeline>) == 4) {
    cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  // Prepare adjustments
  float3 adj0 = image_adjustment.value_or(make_float3(0.0f, 0.0f, 0.0f));
  float3 adj1 = adj0; // Same adjustment for all images in this implementation
  float3 adj2 = adj0;

  if (!stitch_context.is_hard_seam()) {
    // ===================== SOFT SEAM PATH =====================
    // Use fused kernel to directly populate full buffers

    // Clear the full buffers
    cuerr = cudaMemsetAsync(stitch_context.cudaFull0->data(), 0, stitch_context.cudaFull0->size(), stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    cuerr = cudaMemsetAsync(stitch_context.cudaFull1->data(), 0, stitch_context.cudaFull1->size(), stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    cuerr = cudaMemsetAsync(stitch_context.cudaFull2->data(), 0, stitch_context.cudaFull2->size(), stream);
    CUDA_RETURN_IF_ERROR(cuerr);

    // Launch fused remap-to-full kernel
    cuerr = launchFusedRemapToFullKernel3(
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
        canvas_manager,
        adj0,
        adj1,
        adj2,
        image_adjustment.has_value(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);

    SHOW_SCALED(stitch_context.cudaFull0, 0.25);
    SHOW_SCALED(stitch_context.cudaFull1, 0.25);
    SHOW_SCALED(stitch_context.cudaFull2, 0.25);

    // Perform Laplacian blending
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull0;
    cuerr = cudaBatchedLaplacianBlendWithContext3(
        stitch_context.cudaFull0->data_raw(),
        stitch_context.cudaFull1->data_raw(),
        stitch_context.cudaFull2->data_raw(),
        stitch_context.cudaBlendSoftSeam->data_raw(),
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stitch_context.cudaFull0->channels(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);

    SHOW_SCALED(&cudaBlendedFull, 0.25);

    // Copy blended region back to canvas
    assert(canvas_manager._x_blend_start >= 0 && canvas_manager._y_blend_start >= 0);
    cuerr = copy_roi_batched(
        cudaBlendedFull.surface(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->surface(),
        0,
        0,
        stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);

  } else {
    // ===================== HARD SEAM PATH =====================
    // Use single fused kernel to process all three images

    cuerr = launchFusedRemapHardSeam3(
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
        adj0,
        adj1,
        adj2,
        image_adjustment.has_value(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  return std::move(canvas);
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

  if (match_exposure_ && !image_adjustment_.has_value()) {
    image_adjustment_ = compute_image_adjustment(inputImage0, inputImage1, inputImage2);
    if (!image_adjustment_.has_value()) {
      return CudaStatus(cudaError_t::cudaErrorAssert, "Unable to compute 3-image adjustment");
    }
  }

  // Use optimized implementation by default
  auto result = process_impl_optimized(
      inputImage0,
      inputImage1,
      inputImage2,
      *stitch_context_,
      *canvas_manager_,
      image_adjustment_,
      stream,
      std::move(canvas));

  if (stream) {
    cudaStreamSynchronize(stream);
  }

  if (!result.ok()) {
    status_.Update(result.status());
  }

  return result;
}

} // namespace cuda
} // namespace pano
} // namespace hm
