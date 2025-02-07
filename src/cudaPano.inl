#pragma once

#include "cudaImageAdjust.h"
#include "cudaMakeFull.h"
#include "cudaPano.h"
#include "cudaRemap.h"

namespace hm {
namespace pano {
namespace cuda {

template <typename T_pipeline, typename T_compute>
CudaStitchPano<T_pipeline, T_compute>::CudaStitchPano(
    int batch_size,
    int num_levels,
    const ControlMasks& control_masks,
    bool match_exposure)
    : match_exposure_(match_exposure) {
  stitch_context_ = std::make_unique<StitchingContext<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size, /*is_hard_seam=*/num_levels == 0);
  assert(!control_masks.positions.empty());
  // Compute canvas size
  const int canvas_width = std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols);
  const int canvas_height = std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows);

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

constexpr inline float3 neg(const float3& f) {
  return float3{
      .x = f.x,
      .y = f.y,
      .z = f.z,
  };
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    StitchingContext<T_pipeline, T_compute>& stitch_context,
    const CanvasManager& canvas_manager,
    const std::optional<T_compute>& image_adjustment,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CudaStatus cuerr;

  assert(canvas);

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
    cuerr = batched_remap_kernel_ex_offset(
        inputImage1.data(),
        inputImage1.width(),
        inputImage1.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_1_x->data(),
        stitch_context.remap_1_y->data(),
        {0, 0, 0},
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_1_x->width(),
        stitch_context.remap_1_x->height(),
        /*offsetX=*/canvas_manager._x1,
        /*offsetY=*/canvas_manager._y1,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SMALL(canvas);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 1 from the canvas onto the blend image
    //
    cuerr = simple_make_full_batch<BaseScalar_t<T_pipeline>, BaseScalar_t<T_compute>, unsigned char>(
        // Image 1 (float image)
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        /*region_width=*/roi_width(canvas_manager.roi_blend_1),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_1)*/,
        /*channels=*/3,
        // Batch of masks (optional)
        nullptr,
        0,
        0,
        0,
        canvas_manager.roi_blend_1.x,
        0 /* we've already applied our Y offset */,
        /*destOffsetX=*/canvas_manager._remapper_1.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull1->data_raw(),
        /*d_full_masks=*/nullptr,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(stitch_context.cudaFull1);
#endif
  } else {
    //
    // HARD SEAM LEFT
    //
#if 1
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage1.data(),
          inputImage1.width(),
          inputImage1.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
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
          neg(*image_adjustment),
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage1.data(),
          inputImage1.width(),
          inputImage1.height(),
          canvas->data(),
          canvas->width(),
          canvas->height(),
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
    // SHOW_SMALL(&inputImage1);
    // SHOW_IMAGE(canvas);
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
    cuerr = batched_remap_kernel_ex_offset(
        inputImage2.data(),
        inputImage2.width(),
        inputImage2.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_2_x->data(),
        stitch_context.remap_2_y->data(),
        {0, 0, 0},
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_2_x->width(),
        stitch_context.remap_2_x->height(),
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/canvas_manager._y2,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SMALL(canvas);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 2 from the canvas onto the blend image
    //
    // assert(stitch_context.cudaBlendSoftSeam->height() == roi_height(canvas_manager.roi_blend_2));
    cuerr = simple_make_full_batch<BaseScalar_t<T_pipeline>, BaseScalar_t<T_compute>, unsigned char>(
        // Image 1 (float image)
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        /*region_width=*/roi_width(canvas_manager.roi_blend_2),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_2)*/,
        /*channels=*/3,
        // Batch of masks (optional)
        nullptr,
        0,
        0,
        0,
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/canvas_manager._y2,
        /*destOffsetX=*/canvas_manager._remapper_2.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull2->data_raw(),
        /*d_full_masks=*/nullptr,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(stitch_context.cudaFull2);
#endif
  } else {
    //
    // HARD SEAM RIGHT
    //
#if 1
    assert(canvas_manager._x2 + stitch_context.remap_2_x->width() <= canvas->width());
    assert(canvas_manager._y2 + stitch_context.remap_2_x->height() <= canvas->height());
    cuerr = batched_remap_kernel_ex_offset_with_dest_map(
        inputImage2.data(),
        inputImage2.width(),
        inputImage2.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
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
    // SHOW_SMALL(&inputImage2);
    // SHOW_SMALL(canvas);
    // SHOW_SMALL(stitch_context.cudaBlendHardSeam);
#endif
  }
  if (!stitch_context.is_hard_seam()) {
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull1;
#if 1
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
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(&cudaBlendedFull);
#endif

#if 1
    //
    // Copy the blended portion (overlapping portion + some padding) onto
    // the canvas over some of the remapped image 1 and image 2
    //
    cuerr = copyRoiBatchedInterface(
        cudaBlendedFull.data(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->data(),
        canvas->width(),
        canvas->height(),
        /*offsetX=*/canvas_manager._x2 - canvas_manager.overlap_padding(),
        /*offsetY=*/0,
        /*channels=*/1, // <-- 1 when using stuff like float3
        /*batchSize=*/stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(canvas);
#endif
  }
  return std::move(canvas);
}

template <typename T_pipeline, typename T_compute>
std::optional<float3> CudaStitchPano<T_pipeline, T_compute>::compute_image_adjustment(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2) {
  std::optional<cv::Scalar> adjustment_result = match_seam_images(
      inputImage1.download(),
      inputImage2.download(),
      *whole_seam_mask_image_,
      /*N=*/6,
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
  return process(inputImage1, inputImage2, *stitch_context_, *canvas_manager_, std::nullopt, stream, std::move(canvas));
}

} // namespace cuda
} // namespace pano
} // namespace hm
