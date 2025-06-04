#pragma once

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/imgproc.hpp>
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/utils/cudaBlendShow.h"
#include "cupano/utils/showImage.h" /*NOLINT*/

#include <csignal>
#include <optional>

namespace hm {
namespace pano {
namespace cuda {

/**
 * Constructor (same pattern as the 2‐image version, but now for THREE images).
 * - Loads three remap‐x/y TIFFs from control_masks, and a 3‐channel “soft seam” mask
 *   (or a single‐channel “hard seam” if num_levels==0).
 * - Builds a CanvasManager3 from three positions.
 * - Allocates cudaFull0, cudaFull1, cudaFull2 if soft‐seam.
 */
template <typename T_pipeline, typename T_compute>
CudaStitchPano3<T_pipeline, T_compute>::CudaStitchPano3(
    int batch_size,
    int num_levels,
    const ControlMasks3& control_masks,
    bool match_exposure,
    bool quiet)
    : match_exposure_(match_exposure) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaError_t::cudaErrorFileNotFound, "Stitching masks (3‐image) were not able to be loaded");
    return;
  }
  // 1) Create stitch_context:
  stitch_context_ = std::make_unique<StitchingContext3<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size,
      /*is_hard_seam=*/(num_levels == 0));

  // 2) CanvasManager3:
  assert(control_masks.positions.size() == 3);
  const int canvas_w = static_cast<int>(control_masks.canvas_width());
  const int canvas_h = static_cast<int>(control_masks.canvas_height());

  if (!quiet) {
    std::cout << "Stitched (3‐image) canvas size: " << canvas_w << " x " << canvas_h << std::endl;
  }

  canvas_manager_ = std::make_unique<CanvasManager3>(
      CanvasInfo{
          .width = canvas_w,
          .height = canvas_h,
          .positions =
              {cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
               cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos),
               cv::Point(control_masks.positions[2].xpos, control_masks.positions[2].ypos)}},
      /*minimize_blend=*/!stitch_context_->is_hard_seam());

  // Remapping image sizes:
  canvas_manager_->_remapper_0.width = control_masks.img0_col.cols;
  canvas_manager_->_remapper_0.height = control_masks.img0_col.rows;
  canvas_manager_->_remapper_1.width = control_masks.img1_col.cols;
  canvas_manager_->_remapper_1.height = control_masks.img1_col.rows;
  canvas_manager_->_remapper_2.width = control_masks.img2_col.cols;
  canvas_manager_->_remapper_2.height = control_masks.img2_col.rows;

  canvas_manager_->updateMinimizeBlend(
      control_masks.img0_col.size(), control_masks.img1_col.size(), control_masks.img2_col.size());

  // Load the seam mask (3‐channel) if soft‐seam, else load single‐channel:
  cv::Mat seam_indexed = control_masks.whole_seam_mask_image; // CV_8UC3 if soft-seam
  assert(seam_indexed.type() == CV_8UC1);

  if (!stitch_context_->is_hard_seam()) {
    int n_channels = sizeof(T_compute) / sizeof(BaseScalar_t<T_compute>);
    cv::Mat seam_color = make_n_channel_seam_image(seam_indexed, n_channels);
    // Convert to T_compute type (float, etc.) but keep 3 channels
    seam_color.convertTo(seam_color, cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value));
    // Allocate cudaFull0/1/2 with the seam dimensions:
    stitch_context_->cudaFull0 = std::make_unique<CudaMat<T_compute>>(batch_size, seam_color.cols, seam_color.rows);
    stitch_context_->cudaFull1 = std::make_unique<CudaMat<T_compute>>(batch_size, seam_color.cols, seam_color.rows);
    stitch_context_->cudaFull2 = std::make_unique<CudaMat<T_compute>>(batch_size, seam_color.cols, seam_color.rows);

    stitch_context_->cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(seam_color);
    stitch_context_->laplacian_blend_context =
        std::make_unique<CudaBatchLaplacianBlendContext3<BaseScalar_t<T_compute>>>(
            seam_color.cols,
            seam_color.rows,
            num_levels,
            /*batch_size=*/batch_size);
  } else {
    // Hard-seam: single channel
    assert(seam_indexed.type() == CV_8UC1);
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_indexed);
  }

  if (match_exposure_) {
    whole_seam_mask_image_ = control_masks.whole_seam_mask_image;
  }

  // Now load the remappers into context:
  assert(control_masks.img0_col.type() == CV_16U);
  stitch_context_->remap_0_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img0_col);
  stitch_context_->remap_0_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img0_row);
  stitch_context_->remap_1_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_col);
  stitch_context_->remap_1_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_row);
  stitch_context_->remap_2_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_col);
  stitch_context_->remap_2_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_row);
}

namespace tmp3 {
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

} // namespace tmp3

/**
 * The “core” process_impl for THREE images:
 *  - Remap image0/1/2 onto the canvas (soft or hard seam).
 *  - If soft seam: copy the overlapping/cropped region out of the canvas into cudaFull0/1/2.
 *  - Call cudaBatchedLaplacianBlendWithContext3(...) on (cudaFull0, cudaFull1, cudaFull2) + 3-channel mask,
 *    writing into cudaFull0.
 *  - Copy the blended region (cudaFull0) back onto the canvas.
 */
template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process_impl(
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

  // SHOW_IMAGE(&inputImage0);
  // SHOW_IMAGE(&inputImage1);
  // SHOW_IMAGE(&inputImage2);

  // We need all alphas to be zero to start
  if constexpr (sizeof(T_pipeline) / sizeof(BaseScalar_t<T_pipeline>) == 4) {
    cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  // -------------------- IMAGE 0 --------------------
  if (!stitch_context.is_hard_seam()) {
    // SOFT-SEAM: remap image0 onto canvas
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_adjust(
          inputImage0.surface(),
          canvas->surface(),
          stitch_context.remap_0_x->data(),
          stitch_context.remap_0_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_0_x->width(),
          stitch_context.remap_0_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[0].x,
          /*offsetY=*/canvas_manager.canvas_positions()[0].y,
          /*no_unmapped_write=*/false,
          tmp3::neg(*image_adjustment),
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset(
          inputImage0.surface(),
          canvas->surface(),
          stitch_context.remap_0_x->data(),
          stitch_context.remap_0_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_0_x->width(),
          stitch_context.remap_0_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[0].x,
          /*offsetY=*/canvas_manager.canvas_positions()[0].y,
          /*no_unmapped_write=*/false,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);

    // Copy the region of canvas that is flagged for blending (ROI0) into cudaFull0:
    cuerr = simple_make_full_batch(
        canvas->surface(),
        /*region_width=*/canvas_manager.remapped_image_roi_blend_0.width,
        /*region_height=*/canvas_manager.remapped_image_roi_blend_0.height,
        /*offsetX=*/canvas_manager.canvas_positions()[0].x,
        /*offsetY=*/canvas_manager.canvas_positions()[0].y,
        /*destOffsetX=*/canvas_manager._remapper_0.xpos,
        /*destOffsetY=*/0,
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull0->surface(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  } else {
    // HARD-SEAM for image0 (first of three): use remap_ex_offset_with_dest_map, using mask channel 0
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage0.surface(),
          canvas->surface(),
          stitch_context.remap_0_x->data(),
          stitch_context.remap_0_y->data(),
          {
              0,
          },
          /*this_image_index=*/0, // channel 0 in the 3-channel mask
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_0_x->width(),
          stitch_context.remap_0_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[0].x,
          /*offsetY=*/canvas_manager.canvas_positions()[0].y,
          tmp3::neg(*image_adjustment),
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage0.surface(),
          canvas->surface(),
          stitch_context.remap_0_x->data(),
          stitch_context.remap_0_y->data(),
          {
              0,
          },
          /*this_image_index=*/0,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_0_x->width(),
          stitch_context.remap_0_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[0].x,
          /*offsetY=*/canvas_manager.canvas_positions()[0].y,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  // SHOW_IMAGE(canvas);
  // SHOW_IMAGE(stitch_context.cudaFull0);

  // -------------------- IMAGE 1 --------------------
  if (!stitch_context.is_hard_seam()) {
    // SOFT-SEAM: remap image1 onto canvas
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_adjust(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[1].x,
          /*offsetY=*/canvas_manager.canvas_positions()[1].y,
          /*no_unmapped_write=*/false,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[1].x,
          /*offsetY=*/canvas_manager.canvas_positions()[1].y,
          /*no_unmapped_write=*/false,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);

    // Copy blending region (ROI1) into cudaFull1
    cuerr = simple_make_full_batch(
        canvas->surface(),
        /*region_width=*/canvas_manager.remapped_image_roi_blend_1.width,
        /*region_height=*/canvas_manager.remapped_image_roi_blend_1.height,
        /*offsetX=*/canvas_manager.canvas_positions()[1].x,
        /*offsetY=*/canvas_manager.canvas_positions()[1].y,
        /*destOffsetX=*/canvas_manager._remapper_1.xpos,
        /*destOffsetY=*/0,
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull1->surface(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  } else {
    // HARD-SEAM for image1: channel=1 in mask
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {
              0,
          },
          /*this_image_index=*/1,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[1].x,
          /*offsetY=*/canvas_manager.canvas_positions()[1].y,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage1.surface(),
          canvas->surface(),
          stitch_context.remap_1_x->data(),
          stitch_context.remap_1_y->data(),
          {
              0,
          },
          /*this_image_index=*/1,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_1_x->width(),
          stitch_context.remap_1_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[1].x,
          /*offsetY=*/canvas_manager.canvas_positions()[1].y,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  // SHOW_IMAGE(canvas);
  // SHOW_IMAGE(stitch_context.cudaFull1);

  // -------------------- IMAGE 2 --------------------
  if (!stitch_context.is_hard_seam()) {
    // SOFT-SEAM: remap image2 onto canvas
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_adjust(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[2].x,
          /*offsetY=*/canvas_manager.canvas_positions()[2].y,
          /*no_unmapped_write=*/false,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {
              0,
          },
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[2].x,
          /*offsetY=*/canvas_manager.canvas_positions()[2].y,
          /*no_unmapped_write=*/false,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);

    // Copy blending region (ROI2) into cudaFull2
    cuerr = simple_make_full_batch(
        canvas->surface(),
        /*region_width=*/canvas_manager.remapped_image_roi_blend_2.width,
        /*region_height=*/canvas_manager.remapped_image_roi_blend_2.height,
        /*offsetX=*/canvas_manager.canvas_positions()[2].x,
        /*offsetY=*/canvas_manager.canvas_positions()[2].y,
        /*destOffsetX=*/canvas_manager._remapper_2.xpos,
        /*destOffsetY=*/0,
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull2->surface(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  } else {
    // HARD-SEAM for image2: channel=2 in mask
    if (image_adjustment.has_value()) {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map_adjust(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {
              0,
          },
          /*this_image_index=*/2,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[2].x,
          /*offsetY=*/canvas_manager.canvas_positions()[2].y,
          *image_adjustment,
          stream);
    } else {
      cuerr = batched_remap_kernel_ex_offset_with_dest_map(
          inputImage2.surface(),
          canvas->surface(),
          stitch_context.remap_2_x->data(),
          stitch_context.remap_2_y->data(),
          {
              0,
          },
          /*this_image_index=*/2,
          stitch_context.cudaBlendHardSeam->data(),
          /*batchSize=*/stitch_context.batch_size(),
          stitch_context.remap_2_x->width(),
          stitch_context.remap_2_x->height(),
          /*offsetX=*/canvas_manager.canvas_positions()[2].x,
          /*offsetY=*/canvas_manager.canvas_positions()[2].y,
          stream);
    }
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  // SHOW_IMAGE(canvas);
  // SHOW_IMAGE(stitch_context.cudaFull2);

  // --------------- BLEND (Soft‐Seam) ---------------
  if (!stitch_context.is_hard_seam()) {
    // We now have three “full” images in cudaFull0, cudaFull1, cudaFull2,
    // each of size (mask_width × mask_height) in T_compute type.
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull0;

    // SHOW_IMAGE(stitch_context.cudaFull0);
    // SHOW_IMAGE(stitch_context.cudaFull1);
    // SHOW_IMAGE(stitch_context.cudaFull2);

    cuerr = cudaBatchedLaplacianBlendWithContext3(
        stitch_context.cudaFull0->data_raw(),
        stitch_context.cudaFull1->data_raw(),
        stitch_context.cudaFull2->data_raw(),
        stitch_context.remap_0_x->data(),
        stitch_context.remap_0_y->data(),
        stitch_context.remap_1_x->data(),
        stitch_context.remap_1_y->data(),
        stitch_context.remap_2_x->data(),
        stitch_context.remap_2_y->data(),
        stitch_context.cudaBlendSoftSeam->data_raw(),
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stitch_context.cudaFull0->channels(), // num channels = 3 or 4
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);

    // cuerr = cudaBatchedLaplacianBlend3(
    //     stitch_context.cudaFull0->data_raw(),
    //     stitch_context.cudaFull1->data_raw(),
    //     stitch_context.cudaFull2->data_raw(),
    //     stitch_context.cudaBlendSoftSeam->data_raw(),
    //     cudaBlendedFull.data_raw(),
    //     stitch_context.cudaFull0->width(),
    //     stitch_context.cudaFull0->height(),
    //     stitch_context.cudaFull0->channels(),
    //     /*numLevels=*/6,
    //     /*batchSize=*/1,
    //     stream);
    // CUDA_RETURN_IF_ERROR(cuerr);

    // SHOW_IMAGE(stitch_context.cudaFull0);
    // SHOW_IMAGE(stitch_context.cudaFull1);
    // SHOW_IMAGE(stitch_context.cudaFull2);

    // stitch_context.laplacian_blend_context->displayPyramids(
    //     /*channels=*/stitch_context.cudaFull0->channels(), /*scale=*/1.0, /*wait=*/true);

    // SHOW_IMAGE(stitch_context.cudaFull0);
    // SHOW_IMAGE(stitch_context.cudaFull1);
    // SHOW_IMAGE(stitch_context.cudaFull2);
    // SHOW_IMAGE(canvas);
    // SHOW_IMAGE(&cudaBlendedFull);

    assert(canvas_manager._x_blend_start >= 0 && canvas_manager._y_blend_start >= 0);

    // Copy the blended region back onto the canvas:
    cuerr = copy_roi_batched(
        cudaBlendedFull.surface(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->surface(),
        /*offsetX=*/canvas_manager._x_blend_start,
        /*offsetY=*/canvas_manager._y_blend_start,
        /*batchSize=*/stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
  }

  return std::move(canvas);
}

/**
 * Top‐level “process”:
 *  - If match_exposure_ is set, compute a 3‐channel offset from three inputs + 3‐channel seam.
 *  - Call process_impl(...)
 *  - Sync stream, update status if needed.
 */
template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CUDA_RETURN_IF_ERROR(status_);
  if (match_exposure_ && !image_adjustment_.has_value()) {
    image_adjustment_ = compute_image_adjustment(inputImage0, inputImage1, inputImage2);
    if (!image_adjustment_.has_value()) {
      return CudaStatus(cudaError_t::cudaErrorAssert, "Unable to compute 3‐image adjustment");
    }
  }
  auto result = process_impl(
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

template <typename T_pipeline, typename T_compute>
std::optional<float3> CudaStitchPano3<T_pipeline, T_compute>::compute_image_adjustment(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<T_pipeline>& inputImage3) {
  // cv::Mat tmp1 = inputImage1.download();
  // cv::Mat tmp2 = inputImage2.download();
  // std::optional<cv::Scalar> adjustment_result = match_seam_images(
  //     tmp1,
  //     tmp2,
  //     *whole_seam_mask_image_,
  //     /*N=*/100,
  //     canvas_manager_->canvas_positions()[0],
  //     canvas_manager_->canvas_positions()[1]);
  // if (adjustment_result.has_value()) {
  //   const cv::Scalar& adjustment = *adjustment_result;
  //   return float3{
  //       .x = (float)adjustment[0],
  //       .y = (float)adjustment[1],
  //       .z = (float)adjustment[2],
  //   };
  // }
  return std::nullopt;
}
/**
 * A naive, direct extension of “match_seam_images” that now takes THREE images and a 3‐channel seam.
 * We sample around the seam boundary for each image’s region, accumulate per‐channel sums in image0, image1, image2,
 * then compute a shared offset.  For brevity, we simply average all three sets of pixels and compute
 * a single 3‐vector to apply to all three.
 * (Implementation omitted—same logic as match_seam_images, but extended to three.)
 */
template <typename T_pipeline, typename T_compute>
std::optional<cv::Scalar> match_seam_images3(
    cv::Mat& image0,
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam_color,
    int N,
    const cv::Point& topLeft0,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose) {
  // For brevity, imagine we partition the seam_color into three single‐channel masks:
  // ch0, ch1, ch2.  Then sample from “inside” each image similarly to match_seam_images,
  // accumulate sums_0, sums_1, sums_2.  Then produce a single cv::Scalar offset3 = (sums_0/ct0 + sums_1/ct1 +
  // sums_2/ct2)/3, return offset3 so that image0 -= offset3, image1 += offset3, image2 += offset3. (Full code would
  // mirror the two‐image version.)
  //
  // … [Implementation Omitted Here for Brevity] …
  //
  return std::nullopt;
}

template <typename T_pipeline, typename T_compute>
cv::Mat CudaStitchPano3<T_pipeline, T_compute>::make_n_channel_seam_image(const cv::Mat& seam_image, int n_channels) {
  assert(seam_image.type() == CV_8UC1);
  // 2) Find the maximum label N (so we know how many output channels to allocate):

  // 3) Prepare a vector of single‐channel masks, one per label 0..N:
  std::vector<cv::Mat> masks;
  masks.reserve(n_channels);

  for (int k = 0; k < n_channels; ++k) {
    // (seam == k) produces a CV_8U mask with 255 where seam==k, else 0.
    cv::Mat binMask = (seam_image == k);

    // Convert from {0,255} → {0,1} by dividing by 255.
    // (if you prefer CV_32F, use binMask.convertTo(binMask, CV_32F) / 255.f)
    binMask /= 255;

    // Now binMask is CV_8U with exactly 0 or 1.
    masks.push_back(binMask);
  }

  // 4) Merge all single‐channel masks into one multi‐channel image:
  //    This will create a CV_8UC(N+1) Mat of size same as `seam`.
  cv::Mat oneHot;
  cv::merge(masks, oneHot);

  // oneHot.type() == CV_8UC{N+1}.  For example, if N=2, then CV_8UC3.
  // At (y,x):
  //   oneHot.at<Vec<uchar,3>>(y,x)[0] == 1 iff seam(y,x)==0
  //   oneHot.at<Vec<uchar,3>>(y,x)[1] == 1 iff seam(y,x)==1
  //   oneHot.at<Vec<uchar,3>>(y,x)[2] == 1 iff seam(y,x)==2
  return oneHot;
}

} // namespace cuda
} // namespace pano
} // namespace hm
