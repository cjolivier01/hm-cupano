#pragma once

#include <opencv2/imgproc.hpp>
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano.h"
#include "cupano/utils/cudaBlendShow.h"
#include "cupano/utils/showImage.h" /*NOLINT*/

#include <algorithm>
#include <cmath>
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
    bool quiet,
    bool minimize_blend,
    int max_output_width)
    : match_exposure_(match_exposure) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaError_t::cudaErrorFileNotFound, "Stitching masks were not able to be loaded");
    return;
  }
  stitch_context_ = std::make_unique<StitchingContext<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size, /*is_hard_seam=*/num_levels == 0);
  assert(control_masks.positions.size() == 2);

  const int orig_canvas_width = static_cast<int>(control_masks.canvas_width());
  const int orig_canvas_height = static_cast<int>(control_masks.canvas_height());

  auto pad_mask_to_canvas = [](const cv::Mat& mask, int canvas_w, int canvas_h) -> cv::Mat {
    int padw = std::max(0, canvas_w - mask.cols);
    int padh = std::max(0, canvas_h - mask.rows);
    if (padw > 0 || padh > 0) {
      cv::Mat padded;
      cv::copyMakeBorder(mask, padded, 0, padh, 0, padw, cv::BORDER_REPLICATE);
      assert(padded.cols == canvas_w);
      assert(padded.rows == canvas_h);
      return padded;
    }
    return mask;
  };

  // Cache the original seam + positions for exposure matching. These must stay
  // in the original coordinate system even when we downscale the pano output.
  cv::Mat seam_mask_padded =
      pad_mask_to_canvas(control_masks.whole_seam_mask_image, orig_canvas_width, orig_canvas_height);
  if (match_exposure_) {
    whole_seam_mask_image_ = seam_mask_padded.clone();
    exposure_positions_ = std::array<cv::Point, 2>{
        cv::Point(static_cast<int>(control_masks.positions[0].xpos), static_cast<int>(control_masks.positions[0].ypos)),
        cv::Point(static_cast<int>(control_masks.positions[1].xpos), static_cast<int>(control_masks.positions[1].ypos)),
    };
  }

  // Compute canvas size, optionally downscaling everything up-front so the CUDA
  // kernels run at the requested output resolution (avoids an expensive post-resize).
  int canvas_width = orig_canvas_width;
  int canvas_height = orig_canvas_height;
  float scale = 1.0f;
  if (max_output_width > 0 && orig_canvas_width > max_output_width) {
    canvas_width = max_output_width;
    if (canvas_width % 2 != 0) {
      canvas_width -= 1;
    }
    canvas_width = std::max(2, canvas_width);
    scale = static_cast<float>(canvas_width) / static_cast<float>(orig_canvas_width);
    canvas_height = static_cast<int>(orig_canvas_height * scale);
    if (canvas_height % 2 != 0) {
      canvas_height -= 1;
    }
    canvas_height = std::max(2, canvas_height);
  }

  // Locals that may be resized when scale != 1.0
  cv::Mat img1_col = control_masks.img1_col;
  cv::Mat img1_row = control_masks.img1_row;
  cv::Mat img2_col = control_masks.img2_col;
  cv::Mat img2_row = control_masks.img2_row;
  cv::Mat seam_mask = seam_mask_padded;

  std::array<cv::Point, 2> canvas_positions{
      cv::Point(static_cast<int>(control_masks.positions[0].xpos), static_cast<int>(control_masks.positions[0].ypos)),
      cv::Point(static_cast<int>(control_masks.positions[1].xpos), static_cast<int>(control_masks.positions[1].ypos)),
  };

  if (scale != 1.0f) {
    auto scale_point = [&](const SpatialTiff& pos) -> cv::Point {
      return cv::Point(static_cast<int>(pos.xpos * scale), static_cast<int>(pos.ypos * scale));
    };
    canvas_positions = {scale_point(control_masks.positions[0]), scale_point(control_masks.positions[1])};

    auto clamp_resize = [&](const cv::Mat& src, const cv::Point& pos) -> cv::Size {
      int w = std::max(1, static_cast<int>(src.cols * scale));
      int h = std::max(1, static_cast<int>(src.rows * scale));
      int max_w = std::max(1, canvas_width - pos.x);
      int max_h = std::max(1, canvas_height - pos.y);
      w = std::min(w, max_w);
      h = std::min(h, max_h);
      return cv::Size(w, h);
    };

    const cv::Size s1 = clamp_resize(img1_col, canvas_positions[0]);
    const cv::Size s2 = clamp_resize(img2_col, canvas_positions[1]);

    cv::resize(img1_col, img1_col, s1, 0, 0, cv::INTER_NEAREST);
    cv::resize(img1_row, img1_row, s1, 0, 0, cv::INTER_NEAREST);
    cv::resize(img2_col, img2_col, s2, 0, 0, cv::INTER_NEAREST);
    cv::resize(img2_row, img2_row, s2, 0, 0, cv::INTER_NEAREST);
    cv::resize(seam_mask, seam_mask, cv::Size(canvas_width, canvas_height), 0, 0, cv::INTER_NEAREST);
  }

  if (!quiet) {
    std::cout << "Stitched canvas size: " << canvas_width << " x " << canvas_height << std::endl;
  }

  //
  // CanvasManager
  //
  const bool enable_minimize_blend = minimize_blend && !stitch_context_->is_hard_seam();
  canvas_manager_ = std::make_unique<CanvasManager>(
      CanvasInfo{
          .width = canvas_width, .height = canvas_height, .positions = {canvas_positions[0], canvas_positions[1]}},
      /*minimize_blend=*/enable_minimize_blend);

  canvas_manager_->_remapper_1.width = img1_col.cols;
  canvas_manager_->_remapper_1.height = img1_col.rows;
  canvas_manager_->_remapper_2.width = img2_col.cols;
  canvas_manager_->_remapper_2.height = img2_col.rows;

  canvas_manager_->updateMinimizeBlend(img1_col.size(), img2_col.size());

  cv::Mat blend_seam = canvas_manager_->convertMaskMat(seam_mask);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  assert(img1_col.type() == CV_16U);
  stitch_context_->remap_1_x = std::make_unique<CudaMat<uint16_t>>(img1_col);
  stitch_context_->remap_1_y = std::make_unique<CudaMat<uint16_t>>(img1_row);

  stitch_context_->remap_2_x = std::make_unique<CudaMat<uint16_t>>(img2_col);
  stitch_context_->remap_2_y = std::make_unique<CudaMat<uint16_t>>(img2_row);

  if (!stitch_context_->is_hard_seam()) {
    blend_seam.convertTo(blend_seam, cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value));
    stitch_context_->cudaFull1 =
        std::make_unique<CudaMat<T_compute>>(stitch_context_->batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context_->cudaFull2 =
        std::make_unique<CudaMat<T_compute>>(stitch_context_->batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context_->cudaBlended =
        std::make_unique<CudaMat<T_compute>>(stitch_context_->batch_size(), blend_seam.cols, blend_seam.rows);

    stitch_context_->cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(blend_seam);
    stitch_context_->laplacian_blend_context =
        std::make_unique<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>>(
            stitch_context_->cudaBlendSoftSeam->width(),
            stitch_context_->cudaBlendSoftSeam->height(),
            num_levels,
            /*batch_size=*/stitch_context_->batch_size());

    // One-time clear: the per-frame ROI copies intentionally do not touch the padding/empty
    // regions of these full-frame blend buffers.
    const size_t full1_bytes = static_cast<size_t>(stitch_context_->cudaFull1->pitch()) *
        static_cast<size_t>(stitch_context_->cudaFull1->height()) *
        static_cast<size_t>(stitch_context_->cudaFull1->batch_size());
    cudaError_t clear_err = cudaMemset(stitch_context_->cudaFull1->data(), 0, full1_bytes);
    if (clear_err != cudaSuccess) {
      status_ = CudaStatus(clear_err, "Failed to clear blend buffer cudaFull1");
      return;
    }
    const size_t full2_bytes = static_cast<size_t>(stitch_context_->cudaFull2->pitch()) *
        static_cast<size_t>(stitch_context_->cudaFull2->height()) *
        static_cast<size_t>(stitch_context_->cudaFull2->batch_size());
    clear_err = cudaMemset(stitch_context_->cudaFull2->data(), 0, full2_bytes);
    if (clear_err != cudaSuccess) {
      status_ = CudaStatus(clear_err, "Failed to clear blend buffer cudaFull2");
      return;
    }
  } else {
    assert(blend_seam.type() == CV_8U);
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(blend_seam);
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
  bool clear_canvas = true;
  if (!stitch_context.is_hard_seam()) {
    // If we're doing a full-canvas blend, the final ROI copy overwrites every pixel.
    if (!canvas_manager.minimize_blend()) {
      clear_canvas = false;
    } else {
      // When the two remap rectangles fully cover the output canvas, every pixel is written
      // by at least one remap kernel (including unmapped pixels via the default fill).
      const bool covers_full_height = (canvas_manager._y1 == 0 && canvas_manager._y2 == 0 &&
          canvas_manager._remapper_1.height == canvas->height() && canvas_manager._remapper_2.height == canvas->height());
      const bool covers_full_width = (canvas_manager._x1 == 0 && canvas_manager._x2 <= canvas_manager._remapper_1.width &&
          (canvas_manager._x2 + canvas_manager._remapper_2.width) >= canvas->width());
      clear_canvas = !(covers_full_height && covers_full_width);
    }
  }
  if (clear_canvas) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        canvas->data_raw(),
        0,
        canvas->pitch() * canvas->height() * stitch_context.batch_size(),
        stream));
  }

  // TODO: remove me
  // CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull1->data_raw(), 0, stitch_context.cudaFull1->pitch() *
  // stitch_context.cudaFull1->height(), stream));
  // CUDA_RETURN_IF_ERROR(cudaMemsetAsync(stitch_context.cudaFull2->data_raw(), 0, stitch_context.cudaFull2->pitch() *
  // stitch_context.cudaFull2->height(), stream));

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
    cuerr = copy_roi_batched(
        canvas->surface(),
        /*regionWidth=*/roi_width(canvas_manager.remapped_image_roi_blend_1),
        /*regionHeight=*/stitch_context.cudaBlendSoftSeam->height(),
        /*srcROI_x=*/canvas_manager.remapped_image_roi_blend_1.x,
        /*srcROI_y=*/0 /* we've already applied our Y offset */,
        stitch_context.cudaFull1->surface(),
        /*offsetX=*/canvas_manager._remapper_1.xpos,
        /*offsetY=*/0,
        /*batchSize=*/batch_size,
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
    // SHOW_SCALED(canvas, 0.5);
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
    cuerr = copy_roi_batched(
        canvas->surface(),
        /*regionWidth=*/roi_width(canvas_manager.remapped_image_roi_blend_2),
        /*regionHeight=*/stitch_context.cudaBlendSoftSeam->height(),
        /*srcROI_x=*/canvas_manager._x2,
        /*srcROI_y=*/0, // we've already applied the Y offset when painting it onto the canvas
        stitch_context.cudaFull2->surface(),
        /*offsetX=*/canvas_manager._remapper_2.xpos,
        /*offsetY=*/0,
        /*batchSize=*/stitch_context.batch_size(),
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
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaBlended;
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
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stitch_context.cudaFull2->channels(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(&cudaBlendedFull);
    // stitch_context.laplacian_blend_context->displayPyramids(tmp::num_channels<T_compute>(), 0.25, /*wait=*/true);
#if 1
    //
    // Copy the blended portion (overlapping portion + some padding) onto
    // the canvas over some of the remapped image 1 and image 2
    //
    int out_offset_x = 0;
    if (canvas_manager.minimize_blend()) {
      out_offset_x = canvas_manager._x2 - canvas_manager.overlap_padding();
    }
    cuerr = copy_roi_batched(
        cudaBlendedFull.surface(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->surface(),
        /*offsetX=*/out_offset_x,
        /*offsetY=*/0,
        /*batchSize=*/stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(canvas);
    // SHOW_SCALED(canvas, 0.15);
#endif
  }
  return std::move(canvas);
}

template <typename T_pipeline, typename T_compute>
std::optional<float3> CudaStitchPano<T_pipeline, T_compute>::compute_image_adjustment(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2) {
  cv::Mat tmp1 = inputImage1.download();
  cv::Mat tmp2 = inputImage2.download();
  cv::Point top_left_1 = canvas_manager_->canvas_positions()[0];
  cv::Point top_left_2 = canvas_manager_->canvas_positions()[1];
  if (exposure_positions_.has_value()) {
    top_left_1 = (*exposure_positions_)[0];
    top_left_2 = (*exposure_positions_)[1];
  }
  std::optional<cv::Scalar> adjustment_result = match_seam_images(
      tmp1,
      tmp2,
      *whole_seam_mask_image_,
      /*N=*/100,
      top_left_1,
      top_left_2);
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
  if (!result.ok()) {
    status_.Update(result.status());
  }
  return result;
}

} // namespace cuda
} // namespace pano
} // namespace hm
