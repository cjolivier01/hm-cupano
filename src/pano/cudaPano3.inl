#pragma once

#include <opencv2/core/hal/interface.h>
#include <opencv2/imgproc.hpp>
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
    bool quiet,
    int max_output_width,
    bool minimize_blend)
    : minimize_blend_(minimize_blend && num_levels > 0) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaErrorFileNotFound, "Stitching masks (3‐image) were not able to be loaded");
    return;
  }
  std::optional<ControlMasks3> scaled_control_masks;
  if (max_output_width > 0 && control_masks.canvas_width() > static_cast<size_t>(max_output_width)) {
    scaled_control_masks = control_masks;
    if (!scaled_control_masks->scale_to_max_output_width(max_output_width)) {
      status_ = CudaStatus(cudaErrorInvalidValue, "max_output_width removes one or more 3-image seam classes");
      return;
    }
  }
  const ControlMasks3& masks = scaled_control_masks ? *scaled_control_masks : control_masks;

  // 1) Create stitch_context:
  stitch_context_ = std::make_unique<StitchingContext3<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size,
      /*is_hard_seam=*/(num_levels == 0));

  // 2) CanvasManager3:
  assert(masks.positions.size() == 3);
  const int canvas_w = static_cast<int>(masks.canvas_width());
  const int canvas_h = static_cast<int>(masks.canvas_height());

  if (!quiet) {
    std::cout << "Stitched (3‐image) canvas size: " << canvas_w << " x " << canvas_h << std::endl;
  }

  canvas_manager_ = std::make_unique<CanvasManager3>(
      CanvasInfo{
          .width = canvas_w,
          .height = canvas_h,
          .positions =
              {cv::Point(masks.positions[0].xpos, masks.positions[0].ypos),
               cv::Point(masks.positions[1].xpos, masks.positions[1].ypos),
               cv::Point(masks.positions[2].xpos, masks.positions[2].ypos)}},
      // ROI ownership lives in CudaStitchPano3; CanvasManager3's legacy pairwise crop is not used.
      /*minimize_blend=*/false,
      /*overlap_pad=*/blend_roi::scaled_overlap_padding(control_masks.canvas_width(), canvas_w));

  // Remapping image sizes:
  canvas_manager_->_remapper_0.width = masks.img0_col.cols;
  canvas_manager_->_remapper_0.height = masks.img0_col.rows;
  canvas_manager_->_remapper_1.width = masks.img1_col.cols;
  canvas_manager_->_remapper_1.height = masks.img1_col.rows;
  canvas_manager_->_remapper_2.width = masks.img2_col.cols;
  canvas_manager_->_remapper_2.height = masks.img2_col.rows;

  canvas_manager_->updateMinimizeBlend(masks.img0_col.size(), masks.img1_col.size(), masks.img2_col.size());

  // Load the indexed seam mask. The minimized soft path retains the full mask for its hard-seam
  // baseline and crops only the one-hot blend mask.
  cv::Mat seam_indexed = masks.whole_seam_mask_image;
  assert(seam_indexed.type() == CV_8UC1);

  if (!stitch_context_->is_hard_seam()) {
    cv::Mat seam_index_for_blend = seam_indexed;
    if (minimize_blend_) {
      const blend_roi::Regions regions =
          blend_roi::select_regions(seam_indexed, num_levels, canvas_manager_->overlap_padding());
      write_roi_canvas_ = regions.write;
      blend_roi_canvas_ = regions.blend;
      const std::vector<cv::Point> positions(
          canvas_manager_->canvas_positions().begin(), canvas_manager_->canvas_positions().end());
      const std::vector<cv::Mat> remap_x = {masks.img0_col, masks.img1_col, masks.img2_col};
      const std::vector<cv::Mat> remap_y = {masks.img0_row, masks.img1_row, masks.img2_row};
      if (minimizes_blend() &&
          !blend_roi::hard_baseline_covers_soft_owners_outside_write(
              seam_indexed, positions, remap_x, remap_y, write_roi_canvas_)) {
        write_roi_canvas_ = {};
        blend_roi_canvas_ = {};
      }
      if (minimizes_blend()) {
        stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_indexed);
        seam_index_for_blend = seam_indexed(blend_roi_canvas_);
        const std::array<cv::Size, 3> remap_sizes = {
            masks.img0_col.size(), masks.img1_col.size(), masks.img2_col.size()};
        for (size_t i = 0; i < remap_rois_.size(); ++i) {
          remap_rois_[i] =
              blend_roi::remap_roi(canvas_manager_->canvas_positions()[i], remap_sizes[i], blend_roi_canvas_);
        }
        stitch_context_->minimizes_blend = true;
        stitch_context_->blend_roi_canvas = blend_roi_canvas_;
        stitch_context_->write_roi_canvas = write_roi_canvas_;
        stitch_context_->remap_rois = remap_rois_;
      }
    }

    cv::Mat seam_color = ControlMasks3::split_to_channels(seam_index_for_blend);
    // Convert to T_compute type (float, etc.) but keep 3 channels
    seam_color.convertTo(seam_color, cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value));
    // Allocate cudaFull0/1/2 at the effective blend dimensions.
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
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_indexed);
  }

  // Now load the remappers into context:
  assert(masks.img0_col.type() == CV_16U);
  stitch_context_->remap_0_x = std::make_unique<CudaMat<uint16_t>>(masks.img0_col);
  stitch_context_->remap_0_y = std::make_unique<CudaMat<uint16_t>>(masks.img0_row);
  stitch_context_->remap_1_x = std::make_unique<CudaMat<uint16_t>>(masks.img1_col);
  stitch_context_->remap_1_y = std::make_unique<CudaMat<uint16_t>>(masks.img1_row);
  stitch_context_->remap_2_x = std::make_unique<CudaMat<uint16_t>>(masks.img2_col);
  stitch_context_->remap_2_y = std::make_unique<CudaMat<uint16_t>>(masks.img2_row);
}

namespace tmp3 {
inline float3 neg(const float3& f) {
  return make_float3(-f.x, -f.y, -f.z);
}

template <typename T>
inline constexpr size_t num_channels() {
  static_assert(sizeof(T) / sizeof(BaseScalar_t<T>) != 1);
  return sizeof(T) / sizeof(BaseScalar_t<T>);
}

} // namespace tmp3

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPano3<T_pipeline, T_compute>::remap_to_surface_for_blending(
    const CudaMat<T_pipeline>& inputImage,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    CudaMat<T_compute>& dest_canvas,
    int dest_canvas_x,
    int dest_canvas_y,
    int batch_size,
    cudaStream_t stream) {
  CudaStatus cuerr;
  const T_pipeline default_pixel = T_pipeline{};
  assert(map_x.width() == map_y.width() && map_x.height() == map_y.height());
  // SOFT-SEAM: remap image0 onto canvas
  cuerr = batched_remap_kernel_ex_offset(
      inputImage.surface(),
      dest_canvas.surface(),
      map_x.data(),
      map_y.data(),
      /*deflt=*/default_pixel,
      /*batchSize=*/batch_size,
      map_x.width(),
      map_y.height(),
      /*offsetX=*/dest_canvas_x,
      /*offsetY=*/dest_canvas_y,
      /*no_unmapped_write=*/false,
      stream);
  return cuerr;
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPano3<T_pipeline, T_compute>::remap_to_surface_for_hard_seam(
    const CudaMat<T_pipeline>& inputImage,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    uint8_t canvas_position_image_index,
    const CudaMat<unsigned char>& canvas_position_image_index_map,
    CudaMat<T_pipeline>& dest_canvas,
    int dest_canvas_x,
    int dest_canvas_y,
    int batch_size,
    cudaStream_t stream) {
  CudaStatus cuerr;
  const T_pipeline default_pixel = T_pipeline{};
  assert(map_x.width() == map_y.width() && map_x.height() == map_y.height());

  cuerr = batched_remap_kernel_ex_offset_with_dest_map(
      inputImage.surface(),
      dest_canvas.surface(),
      map_x.data(),
      map_y.data(),
      /*deflt=*/default_pixel,
      /*this_image_index=*/canvas_position_image_index,
      canvas_position_image_index_map.data(),
      /*batchSize=*/batch_size,
      map_x.width(),
      map_x.height(),
      /*offsetX=*/dest_canvas_x,
      /*offsetY=*/dest_canvas_y,
      stream);

  return cuerr;
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process_impl(
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

  const std::array<const CudaMat<T_pipeline>*, 3> inputs = {&inputImage0, &inputImage1, &inputImage2};
  const std::array<const CudaMat<uint16_t>*, 3> remap_x = {
      stitch_context.remap_0_x.get(), stitch_context.remap_1_x.get(), stitch_context.remap_2_x.get()};
  const std::array<const CudaMat<uint16_t>*, 3> remap_y = {
      stitch_context.remap_0_y.get(), stitch_context.remap_1_y.get(), stitch_context.remap_2_y.get()};

  if (stitch_context.is_hard_seam() || stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream));
  }

  const auto render_hard_seam = [&]() -> CudaStatus {
    for (size_t i = 0; i < inputs.size(); ++i) {
      CudaStatus status = remap_to_surface_for_hard_seam(
          *inputs[i],
          *remap_x[i],
          *remap_y[i],
          static_cast<uint8_t>(i),
          *stitch_context.cudaBlendHardSeam,
          *canvas,
          canvas_manager.canvas_positions()[i].x,
          canvas_manager.canvas_positions()[i].y,
          stitch_context.batch_size(),
          stream);
      if (!status.ok())
        return status;
    }
    return CudaStatus::OkStatus();
  };

  if (stitch_context.is_hard_seam()) {
    CUDA_RETURN_IF_ERROR(render_hard_seam());
    return std::move(canvas);
  }

  const std::array<CudaMat<T_compute>*, 3> full = {
      stitch_context.cudaFull0.get(), stitch_context.cudaFull1.get(), stitch_context.cudaFull2.get()};
  for (CudaMat<T_compute>* scratch : full) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(scratch->data(), 0, scratch->size(), stream));
  }
  if (stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(render_hard_seam());
    const T_pipeline default_pixel{};
    for (size_t i = 0; i < inputs.size(); ++i) {
      const blend_roi::RemapRoi& remap_roi = stitch_context.remap_rois[i];
      CUDA_RETURN_IF_ERROR(batched_remap_kernel_ex_offset_roi(
          inputs[i]->surface(),
          full[i]->surface(),
          remap_x[i]->data(),
          remap_y[i]->data(),
          default_pixel,
          stitch_context.batch_size(),
          remap_x[i]->width(),
          remap_x[i]->height(),
          remap_roi.offset_x,
          remap_roi.offset_y,
          remap_roi.roi.x,
          remap_roi.roi.y,
          remap_roi.roi.width,
          remap_roi.roi.height,
          /*no_unmapped_write=*/false,
          stream));
    }
  } else {
    for (size_t i = 0; i < inputs.size(); ++i) {
      CUDA_RETURN_IF_ERROR(remap_to_surface_for_blending(
          *inputs[i],
          *remap_x[i],
          *remap_y[i],
          *full[i],
          canvas_manager.canvas_positions()[i].x,
          canvas_manager.canvas_positions()[i].y,
          stitch_context.batch_size(),
          stream));
    }
  }

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
  const cv::Rect write_roi = stitch_context.minimizes_blend
      ? stitch_context.write_roi_canvas
      : cv::Rect(0, 0, stitch_context.cudaFull0->width(), stitch_context.cudaFull0->height());
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
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process_impl_current(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  return process_impl(
      inputImage0, inputImage1, inputImage2, *stitch_context_, *canvas_manager_, stream, std::move(canvas));
}

/**
 * Top‐level “process”:
 *  - Call process_impl(...)
 *  - Sync stream, update status if needed.
 */
template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano3<T_pipeline, T_compute>::process(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas,
    bool fused) {
  if (fused) {
    return process_optimized(inputImage0, inputImage1, inputImage2, stream, std::move(canvas));
  }

  CUDA_RETURN_IF_ERROR(status_);
  auto result = process_impl_current(inputImage0, inputImage1, inputImage2, stream, std::move(canvas));
  if (!result.ok()) {
    status_.Update(result.status());
  }
  return result;
}

// template <typename T_pipeline, typename T_compute>
// cv::Mat CudaStitchPano3<T_pipeline, T_compute>::make_n_channel_seam_image(const cv::Mat& seam_image, int n_channels)
// {
//   assert(seam_image.type() == CV_8UC1);
//   // 2) Find the maximum label N (so we know how many output channels to allocate):

//   // 3) Prepare a vector of single‐channel masks, one per label 0..N:
//   std::vector<cv::Mat> masks;
//   masks.reserve(n_channels);

//   for (int k = 0; k < n_channels; ++k) {
//     // (seam == k) produces a CV_8U mask with 255 where seam==k, else 0.
//     cv::Mat binMask = (seam_image == k);

//     // Convert from {0,255} → {0,1} by dividing by 255.
//     // (if you prefer CV_32F, use binMask.convertTo(binMask, CV_32F) / 255.f)
//     binMask /= 255;

//     // Now binMask is CV_8U with exactly 0 or 1.
//     masks.push_back(binMask);
//   }

//   // 4) Merge all single‐channel masks into one multi‐channel image:
//   //    This will create a CV_8UC(N+1) Mat of size same as `seam`.
//   cv::Mat oneHot;
//   cv::merge(masks, oneHot);

//   // oneHot.type() == CV_8UC{N+1}.  For example, if N=2, then CV_8UC3.
//   // At (y,x):
//   //   oneHot.at<Vec<uchar,3>>(y,x)[0] == 1 iff seam(y,x)==0
//   //   oneHot.at<Vec<uchar,3>>(y,x)[1] == 1 iff seam(y,x)==1
//   //   oneHot.at<Vec<uchar,3>>(y,x)[2] == 1 iff seam(y,x)==2
//   return oneHot;
// }

} // namespace cuda
} // namespace pano
} // namespace hm
