#pragma once

#include <opencv2/imgproc.hpp>
#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaTypes.h"
#include "cupano/pano/cudaPano.h"

#include <algorithm>
#include <cmath>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>

namespace hm {
namespace pano {
namespace cuda {

template <typename T_pipeline, typename T_compute>
CudaStitchPano<T_pipeline, T_compute>::CudaStitchPano(
    int batch_size,
    int num_levels,
    const ControlMasks& control_masks,
    bool quiet,
    bool minimize_blend,
    int max_output_width)
    : minimize_blend_(minimize_blend && num_levels > 0) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaErrorFileNotFound, "Stitching masks were not able to be loaded");
    return;
  }
  const size_t original_canvas_width = control_masks.canvas_width();
  std::optional<ControlMasks> scaled_control_masks;
  if (max_output_width > 0 && original_canvas_width > static_cast<size_t>(max_output_width)) {
    scaled_control_masks = control_masks;
    if (!scaled_control_masks->scale_to_max_output_width(max_output_width)) {
      status_ = CudaStatus(cudaErrorInvalidValue, "Stitching masks lost a seam class while applying max_output_width");
      return;
    }
  }
  const ControlMasks& masks = scaled_control_masks ? *scaled_control_masks : control_masks;
  stitch_context_ = std::make_unique<StitchingContext<T_pipeline, T_compute>>(
      /*batch_size=*/batch_size, /*is_hard_seam=*/num_levels == 0);
  assert(!masks.positions.empty());
  // Compute canvas size
  const int canvas_width = masks.canvas_width();
  const int canvas_height = masks.canvas_height();

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
              {cv::Point(masks.positions[0].xpos, masks.positions[0].ypos),
               cv::Point(masks.positions[1].xpos, masks.positions[1].ypos)}},
      /*minimize_blend=*/false,
      /*overlap_pad=*/blend_roi::scaled_overlap_padding(original_canvas_width, canvas_width));

  canvas_manager_->_remapper_1.width = masks.img1_col.cols;
  canvas_manager_->_remapper_1.height = masks.img1_col.rows;
  canvas_manager_->_remapper_2.width = masks.img2_col.cols;
  canvas_manager_->_remapper_2.height = masks.img2_col.rows;

  canvas_manager_->updateMinimizeBlend(masks.img1_col.size(), masks.img2_col.size());

  cv::Mat full_seam = canvas_manager_->convertMaskMat(masks.whole_seam_mask_image);
  assert(!full_seam.empty());
  full_seam = full_seam.clone();
  cv::Mat blend_seam = full_seam;

  if (minimize_blend_) {
    const blend_roi::Regions regions =
        blend_roi::select_regions(full_seam, num_levels, canvas_manager_->overlap_padding());
    write_roi_canvas_ = regions.write;
    blend_roi_canvas_ = regions.blend;

    // Two-image seam labels are blend weights: 1 owns image 1 and 0 owns image 2.
    const std::vector<cv::Point> owner_positions = {
        canvas_manager_->canvas_positions()[1], canvas_manager_->canvas_positions()[0]};
    const std::vector<cv::Mat> owner_remap_x = {masks.img2_col, masks.img1_col};
    const std::vector<cv::Mat> owner_remap_y = {masks.img2_row, masks.img1_row};
    if (minimizes_blend() &&
        !blend_roi::hard_baseline_covers_soft_owners_outside_write(
            full_seam, owner_positions, owner_remap_x, owner_remap_y, write_roi_canvas_)) {
      write_roi_canvas_ = {};
      blend_roi_canvas_ = {};
    }

    if (minimizes_blend()) {
      stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(full_seam);
      blend_seam = full_seam(blend_roi_canvas_).clone();
      remap_rois_[0] =
          blend_roi::remap_roi(canvas_manager_->canvas_positions()[0], masks.img1_col.size(), blend_roi_canvas_);
      remap_rois_[1] =
          blend_roi::remap_roi(canvas_manager_->canvas_positions()[1], masks.img2_col.size(), blend_roi_canvas_);
      stitch_context_->minimizes_blend = true;
      stitch_context_->blend_roi_canvas = blend_roi_canvas_;
      stitch_context_->write_roi_canvas = write_roi_canvas_;
      stitch_context_->remap_rois = remap_rois_;
    }
  }

  assert(masks.img1_col.type() == CV_16U);
  stitch_context_->remap_1_x = std::make_unique<CudaMat<uint16_t>>(masks.img1_col);
  stitch_context_->remap_1_y = std::make_unique<CudaMat<uint16_t>>(masks.img1_row);

  stitch_context_->remap_2_x = std::make_unique<CudaMat<uint16_t>>(masks.img2_col);
  stitch_context_->remap_2_y = std::make_unique<CudaMat<uint16_t>>(masks.img2_row);

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
    assert(full_seam.type() == CV_8U);
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(full_seam);
  }
}

namespace tmp {
template <typename T>
inline constexpr size_t num_channels() {
  static_assert(sizeof(T) / sizeof(BaseScalar_t<T>) != 1);
  return sizeof(T) / sizeof(BaseScalar_t<T>);
}

} // namespace tmp

namespace {

template <typename T_compute, typename T_scalar>
CudaStatus write_image_pyramid_level(
    const CudaBatchLaplacianBlendContext<T_scalar>& context,
    const std::vector<T_scalar*>& vec_d_ptrs,
    int level,
    const std::string& directory,
    const std::string& label) {
  if (!vec_d_ptrs.at(level)) {
    return CudaStatus::OkStatus();
  }
  const std::filesystem::path subdir = std::filesystem::path(directory) / label;
  std::error_code ec;
  std::filesystem::create_directories(subdir, ec);
  if (ec) {
    return CudaStatus(cudaErrorUnknown, "Unable to create directory " + subdir.string() + ": " + ec.message());
  }

  hm::CudaMat<T_compute> mat(
      reinterpret_cast<T_compute*>(vec_d_ptrs.at(level)),
      context.batchSize,
      context.widths.at(level),
      context.heights.at(level));
  for (int batch_item = 0; batch_item < context.batchSize; ++batch_item) {
    std::ostringstream filename;
    filename << "level_" << level << "_batch_" << batch_item << ".tiff";
    if (!cv::imwrite((subdir / filename.str()).string(), mat.download(batch_item))) {
      return CudaStatus(cudaErrorUnknown, "Unable to write " + (subdir / filename.str()).string());
    }
  }
  return CudaStatus::OkStatus();
}

template <typename T_scalar>
CudaStatus write_mask_pyramid_level(
    const CudaBatchLaplacianBlendContext<T_scalar>& context,
    const std::vector<T_scalar*>& vec_d_ptrs,
    int level,
    const std::string& directory,
    const std::string& label) {
  if (!vec_d_ptrs.at(level)) {
    return CudaStatus::OkStatus();
  }
  const std::filesystem::path subdir = std::filesystem::path(directory) / label;
  std::error_code ec;
  std::filesystem::create_directories(subdir, ec);
  if (ec) {
    return CudaStatus(cudaErrorUnknown, "Unable to create directory " + subdir.string() + ": " + ec.message());
  }

  hm::CudaMat<T_scalar> mat(vec_d_ptrs.at(level), 1, context.widths.at(level), context.heights.at(level));
  std::ostringstream filename;
  filename << "level_" << level << ".tiff";
  if (!cv::imwrite((subdir / filename.str()).string(), mat.download())) {
    return CudaStatus(cudaErrorUnknown, "Unable to write " + (subdir / filename.str()).string());
  }
  return CudaStatus::OkStatus();
}

} // namespace

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process_impl(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    StitchingContext<T_pipeline, T_compute>& stitch_context,
    const CanvasManager& canvas_manager,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  assert(canvas);
  assert(inputImage1.batch_size() == stitch_context.batch_size());
  assert(inputImage2.batch_size() == stitch_context.batch_size());
  assert(canvas->batch_size() == stitch_context.batch_size());

  const T_pipeline default_pixel{};
  const std::array<const CudaMat<T_pipeline>*, 2> inputs = {&inputImage1, &inputImage2};
  const std::array<const CudaMat<uint16_t>*, 2> remap_x = {
      stitch_context.remap_1_x.get(), stitch_context.remap_2_x.get()};
  const std::array<const CudaMat<uint16_t>*, 2> remap_y = {
      stitch_context.remap_1_y.get(), stitch_context.remap_2_y.get()};

  if (stitch_context.is_hard_seam() || stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream));
  }

  const auto render_hard_seam = [&]() -> CudaStatus {
    constexpr std::array<int, 2> kSeamLabels = {1, 0};
    for (size_t i = 0; i < inputs.size(); ++i) {
      CudaStatus status = batched_remap_kernel_ex_offset_with_dest_map(
          inputs[i]->surface(),
          canvas->surface(),
          remap_x[i]->data(),
          remap_y[i]->data(),
          default_pixel,
          kSeamLabels[i],
          stitch_context.cudaBlendHardSeam->data(),
          stitch_context.batch_size(),
          remap_x[i]->width(),
          remap_x[i]->height(),
          canvas_manager.canvas_positions()[i].x,
          canvas_manager.canvas_positions()[i].y,
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

  const std::array<CudaMat<T_compute>*, 2> full = {stitch_context.cudaFull1.get(), stitch_context.cudaFull2.get()};
  for (CudaMat<T_compute>* scratch : full) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(scratch->data(), 0, scratch->size(), stream));
  }
  if (stitch_context.minimizes_blend) {
    CUDA_RETURN_IF_ERROR(render_hard_seam());
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
      CUDA_RETURN_IF_ERROR(batched_remap_kernel_ex_offset(
          inputs[i]->surface(),
          full[i]->surface(),
          remap_x[i]->data(),
          remap_y[i]->data(),
          default_pixel,
          stitch_context.batch_size(),
          remap_x[i]->width(),
          remap_x[i]->height(),
          canvas_manager.canvas_positions()[i].x,
          canvas_manager.canvas_positions()[i].y,
          /*no_unmapped_write=*/false,
          stream));
    }
  }

  CudaMat<T_compute>& blended = *stitch_context.cudaFull1;
  CUDA_RETURN_IF_ERROR(cudaBatchedLaplacianBlendWithContext(
      stitch_context.cudaFull1->data_raw(),
      stitch_context.cudaFull2->data_raw(),
      stitch_context.cudaBlendSoftSeam->data_raw(),
      blended.data_raw(),
      *stitch_context.laplacian_blend_context,
      stitch_context.cudaFull2->channels(),
      stream));
  const cv::Rect write_roi = stitch_context.minimizes_blend
      ? stitch_context.write_roi_canvas
      : cv::Rect(0, 0, stitch_context.cudaFull1->width(), stitch_context.cudaFull1->height());
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
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process_impl_current(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  return process_impl(inputImage1, inputImage2, *stitch_context_, *canvas_manager_, stream, std::move(canvas));
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPano<T_pipeline, T_compute>::dump_soft_blend_pyramid(
    const std::string& directory,
    cudaStream_t stream) const {
  CUDA_RETURN_IF_ERROR(status_);
  if (!stitch_context_ || stitch_context_->is_hard_seam() || !stitch_context_->laplacian_blend_context) {
    return CudaStatus(cudaErrorInvalidValue, "Soft-blend pyramid is only available when num_levels > 0");
  }
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  std::error_code ec;
  std::filesystem::create_directories(directory, ec);
  if (ec) {
    return CudaStatus(cudaErrorUnknown, "Unable to create directory " + directory + ": " + ec.message());
  }

  const auto& context = *stitch_context_->laplacian_blend_context;
  {
    std::ofstream metadata(std::filesystem::path(directory) / "metadata.txt");
    metadata << "num_levels=" << context.numLevels << "\n";
    metadata << "batch_size=" << context.batchSize << "\n";
    metadata << "channels=" << tmp::num_channels<T_compute>() << "\n";
    for (int level = 0; level < context.numLevels; ++level) {
      metadata << "level_" << level << "=" << context.widths.at(level) << "x" << context.heights.at(level) << "\n";
    }
  }

  for (int level = 0; level < context.numLevels; ++level) {
    CUDA_RETURN_IF_ERROR(write_image_pyramid_level<T_compute>(context, context.d_gauss1, level, directory, "gauss1"));
    CUDA_RETURN_IF_ERROR(write_image_pyramid_level<T_compute>(context, context.d_gauss2, level, directory, "gauss2"));
    CUDA_RETURN_IF_ERROR(write_mask_pyramid_level(context, context.d_maskPyr, level, directory, "mask"));
    CUDA_RETURN_IF_ERROR(write_image_pyramid_level<T_compute>(context, context.d_lap1, level, directory, "lap1"));
    CUDA_RETURN_IF_ERROR(write_image_pyramid_level<T_compute>(context, context.d_lap2, level, directory, "lap2"));
    CUDA_RETURN_IF_ERROR(write_image_pyramid_level<T_compute>(context, context.d_blend, level, directory, "blend"));
    CUDA_RETURN_IF_ERROR(
        write_image_pyramid_level<T_compute>(context, context.d_resonstruct, level, directory, "reconstruct"));
  }
  return CudaStatus::OkStatus();
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPano<T_pipeline, T_compute>::process(
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CUDA_RETURN_IF_ERROR(status_);
  auto result = process_impl_current(inputImage1, inputImage2, stream, std::move(canvas));
  if (!result.ok()) {
    status_.Update(result.status());
  }
  return result;
}

} // namespace cuda
} // namespace pano
} // namespace hm
