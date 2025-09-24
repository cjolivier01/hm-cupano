#pragma once

#include <opencv4/opencv2/imgproc.hpp>
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano.h"
#include "cupano/utils/showImage.h" /*NOLINT*/

#include <csignal>
#include <optional>

namespace hm {
namespace pano {
namespace cuda {

template <typename T_pipeline, typename T_compute>
CudaStitchPano<T_pipeline, T_compute>::CudaStitchPano(
    int batch_size,
    int num_levels,
    const ControlMasks& control_masks,
    bool match_exposure,
    bool quiet)
    : match_exposure_(match_exposure) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaError_t::cudaErrorFileNotFound, "Stitching masks were not able to be loaded");
    return;
  }
  stitch_context_ = std::make_unique<StitchingContext<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size, /*is_hard_seam=*/num_levels == 0);
  assert(!control_masks.positions.empty());
  // Compute canvas size
  const int canvas_width = control_masks.canvas_width();
  const int canvas_height = control_masks.canvas_height();

  if (!quiet) {
    std::cout << "Stitched canvas size: " << canvas_width << " x " << canvas_height << std::endl;
  }

  //
  // CanvasManager
  //
  canvas_manager_ = std::make_unique<CanvasManager>(
      CanvasInfo{
          .width = canvas_width,
          .height = canvas_height,
          .positions =
              {cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
               cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos)}},
      /*minimize_blend=*/!stitch_context_->is_hard_seam());

  canvas_manager_->_remapper_1.width = control_masks.img1_col.cols;
  canvas_manager_->_remapper_1.height = control_masks.img1_col.rows;
  canvas_manager_->_remapper_2.width = control_masks.img2_col.cols;
  canvas_manager_->_remapper_2.height = control_masks.img2_col.rows;

  canvas_manager_->updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  cv::Mat blend_seam = canvas_manager_->convertMaskMat(control_masks.whole_seam_mask_image);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  auto canvas = std::make_unique<CudaMat<T_pipeline>>(
      stitch_context_->batch_size(), canvas_manager_->canvas_width(), canvas_manager_->canvas_height());

  assert(control_masks.img1_col.type() == CV_16U);
  stitch_context_->remap_1_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_col);
  stitch_context_->remap_1_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_row);

  stitch_context_->remap_2_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_col);
  stitch_context_->remap_2_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_row);

  if (!stitch_context_->is_hard_seam()) {
    blend_seam.convertTo(blend_seam, cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value));
    stitch_context_->cudaFull1 =
        std::make_unique<CudaMat<T_compute>>(stitch_context_->batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context_->cudaFull2 =
        std::make_unique<CudaMat<T_compute>>(stitch_context_->batch_size(), blend_seam.cols, blend_seam.rows);

    stitch_context_->cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(blend_seam);
    stitch_context_->laplacian_blend_context =
        std::make_unique<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>>(
            stitch_context_->cudaBlendSoftSeam->width(),
            stitch_context_->cudaBlendSoftSeam->height(),
            num_levels,
            /*batch_size=*/stitch_context_->batch_size());
  } else {
    assert(blend_seam.type() == CV_8U);
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(blend_seam);
  }
  if (match_exposure_) {
    whole_seam_mask_image_ = control_masks.whole_seam_mask_image;
  }
}

namespace tmp {
constexpr inline float3 neg(const float3& f) {
  return float3{
      .x = -f.x,
      .y = -f.y,
      .z = -f.z,
  };
}

template <typename T>
inline constexpr size_t num_channels() {
  static_assert(sizeof(T) / sizeof(BaseScalar_t<T>) != 1);
  return sizeof(T) / sizeof(BaseScalar_t<T>);
}

} // namespace tmp

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process_impl(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    StitchingContext<T_pipeline, T_compute>& stitch_context,
    const CanvasManager& canvas_manager,
    const std::optional<float3>& image_adjustment,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CudaStatus cuerr;

  assert(canvas);
  assert(inputImage1.batch_size() == stitch_context.batch_size());
  assert(inputImage2.batch_size() == stitch_context.batch_size());
  assert(canvas->batch_size() == stitch_context.batch_size());
  const size_t batch_size = stitch_context.batch_size();

  assert(canvas->pitch());
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(canvas->data_raw(), 0, canvas->pitch() * canvas->height(), stream));

  // TODO: remove me
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull1->data_raw(), 0, stitch_context.cudaFull1->pitch() * stitch_context.cudaFull1->height(), stream));
  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull2->data_raw(), 0, stitch_context.cudaFull2->pitch() * stitch_context.cudaFull2->height(), stream));

  // bool cross_pollenate_images = true;
  auto roi_width = [](const cv::Rect2i& roi) { return roi.width; };
  if (!stitch_context.is_hard_seam()) {
    //
    // SOFT SEAM LEFT
    //
#if 1
    //
    // Image 1
    //
    // Remap image 1 onto the canvas
    //
    // SHOW_SCALED(&inputImage1, 0.2);
    // SHOW_SCALED(canvas, 0.2);
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_adjust(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*batchSize=*/batch_size,
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          /*no_unmapped_write=*/false,
          tmp::neg(*image_adjustment),
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*batchSize=*/batch_size,
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          /*no_unmapped_write=*/false,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SCALED_BATCH_ITEM(canvas, 0.2, 0);
    // SHOW_SCALED_BATCH_ITEM(canvas, 0.2, 1);
    // SHOW_SCALED(&inputImage1, 0.2);
    // SHOW_SCALED(canvas, 0.5);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 1 from the canvas onto the blend image
    //
    cuerr = simple_make_full_batch(
        // Image 1 (float image)
        canvas->surface(),
        /*region_width=*/roi_width(canvas_manager.remapped_image_roi_blend_1),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height(),
        canvas_manager.remapped_image_roi_blend_1.x,
        0 /* we've already applied our Y offset */,
        /*destOffsetX=*/canvas_manager._remapper_1.xpos,
        /*destOffsetY=*/0,
        /*adjust_origin=*/false,
        /*batchSize=*/batch_size,
        stitch_context.cudaFull1->surface(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SCALED_BATCH_ITEM(stitch_context.cudaFull1, 0.2, 0);
    // SHOW_SCALED_BATCH_ITEM(stitch_context.cudaFull1, 0.2, 1);
    // SHOW_SCALED(stitch_context.cudaFull1, 0.5);
    // SHOW_IMAGE(stitch_context.cudaFull1);
#endif
  } else {
    //
    // HARD SEAM LEFT
    //
#if 1
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          1 /* <-- we inverted the mask at load-time to make it a weight, so image 0 is actually 1 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          tmp::neg(*image_adjustment),
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          1 /* <-- we inverted the mask at load-time to make it a weight, so image 0 is actually 1 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager._x1,
          /*offsetY=*/canvas_manager._y1,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SMALL(&inputImage1);
    // SHOW_IMAGE(canvas);
    // SHOW_SCALED(canvas, 0.15);
#endif
  }
  //
  // Image 2
  //
  if (!stitch_context.is_hard_seam()) {
    //
    // SOFT SEAM RIGHT
    //
#if 1
    //
    // Remap image 2 directly onto the canvas (will overwrite the overlappign portion of image 1)
    //
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_adjust(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          /*no_unmapped_write=*/false,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          /*no_unmapped_write=*/false,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SCALED(&inputImage2, 0.5);
    // SHOW_SCALED(canvas, 0.5);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 2 from the canvas onto the blend image
    //
    cuerr = simple_make_full_batch(
        // Image 1 (float image)
        canvas->surface(),
        /*region_width=*/roi_width(canvas_manager.remapped_image_roi_blend_2),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_2)*/,
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/0, // we've already applied the Y offset when painting it onto the canvas
        /*destOffsetX=*/canvas_manager._remapper_2.xpos,
        /*destOffsetY=*/0 /* we've already applied our Y offset */,
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull2->surface(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SCALED(stitch_context.cudaFull2, 0.5);
#endif
  } else {
    //
    // HARD SEAM RIGHT
    //
#if 1
    assert(canvas_manager._x2 + stitch_context.remap_2_x->width() <= canvas->width());
    assert(canvas_manager._y2 + stitch_context.remap_2_x->height() <= canvas->height());
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          0 /* <-- we inverted the mask at load-time to make it a weight, so image 1 is actually 0 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {0, 0, 0},
          /*this_image_index=*/
          0 /* <-- we inverted the mask at load-time to make it a weight, so image 1 is actually 0 in the mask */,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager._x2,
          /*offsetY=*/canvas_manager._y2,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SCALED(canvas, 0.5);
    // SHOW_SMALL(stitch_context.cudaBlendHardSeam);
#endif
  }
  if (!stitch_context.is_hard_seam()) {
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull1;
#if 0
    if constexpr (sizeof(T_compute) / sizeof(BaseScalar_t<T_compute>) == 4) {
      auto surf1 = stitch_context.cudaFull1->surface();
      auto surf2 = stitch_context.cudaFull2->surface();
      cuerr = AlphaConditionalCopy(
          surf1,
          surf2,
          /*batchSize=*/stitch_context.batch_size(),
          stream);
      CUDA_RETURN_IF_ERROR(cuerr);
    }
    // SHOW_SCALED(stitch_context.cudaFull1, 0.5);
    // SHOW_SCALED(stitch_context.cudaFull2, 0.5);
#endif
    //
    // BLEND THE IMAGES (overlapping portions + some padding)
    //
    cuerr = cudaBatchedLaplacianBlendWithContext(
        stitch_context.cudaFull1->data_raw(),
        stitch_context.cudaFull2->data_raw(),
        stitch_context.cudaBlendSoftSeam->data_raw(),
        // Put output in full-1 memory
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stitch_context.cudaFull2->channels(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(&cudaBlendedFull);
    // stitch_context.laplacian_blend_context->displayPyramids(tmp::num_channels<T_compute>(), 0.25);
#if 1
    //
    // Copy the blended portion (overlapping portion + some padding) onto
    // the canvas over some of the remapped image 1 and image 2
    //
    cuerr = copy_roi_batched(
        cudaBlendedFull.surface(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->surface(),
        /*offsetX=*/canvas_manager._x2 - canvas_manager.overlap_padding(),
        /*offsetY=*/0,
        /*batchSize=*/stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(canvas);
    // SHOW_SCALED(canvas, 0.15);
#endif
  }
  cudaStreamSynchronize(stream);
  return std::move(canvas);
}

template <typename T_pipeline, typename T_compute>
std::optional<float3> CudaStitchPano<T_pipeline, T_compute>::compute_image_adjustment(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2) {
  cv::Mat tmp1 = inputImage1.download();
  cv::Mat tmp2 = inputImage2.download();
  std::optional<cv::Scalar> adjustment_result = match_seam_images(
      tmp1,
      tmp2,
      *whole_seam_mask_image_,
      /*N=*/100,
      canvas_manager_->canvas_positions()[0],
      canvas_manager_->canvas_positions()[1]);
  if (adjustment_result.has_value()) {
    const cv::Scalar& adjustment = *adjustment_result;
    return float3{
        .x = (float)adjustment[0],
        .y = (float)adjustment[1],
        .z = (float)adjustment[2],
    };
  }
  return std::nullopt;
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CUDA_RETURN_IF_ERROR(status_);
  if (match_exposure_ && !image_adjustment_.has_value()) {
    image_adjustment_ = compute_image_adjustment(inputImage1, inputImage2);
    if (!image_adjustment_.has_value()) {
      return CudaStatus(cudaError_t::cudaErrorAssert, "Unable to compute image adjustment");
    }
  }
  auto result = process_impl(
      inputImage1, inputImage2, *stitch_context_, *canvas_manager_, image_adjustment_, stream, std::move(canvas));
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
