#pragma once

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <optional>

#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/pano/blendRoi.h"

namespace hm {
namespace pano {
namespace cuda {

namespace detailN {
inline float3 neg(const float3& f) {
  return make_float3(-f.x, -f.y, -f.z);
}

template <typename T>
inline constexpr int num_channels_v = sizeof(T) / sizeof(BaseScalar_t<T>);

} // namespace detailN

template <typename T_pipeline, typename T_compute>
CudaStitchPanoN<T_pipeline, T_compute>::CudaStitchPanoN(
    int batch_size,
    int num_levels,
    const ControlMasksN& control_masks,
    bool minimize_blend,
    bool quiet,
    int max_output_width)
    : minimize_blend_(minimize_blend && num_levels > 0) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaErrorFileNotFound, "Stitching masks (N-image) could not be loaded");
    return;
  }

  std::optional<ControlMasksN> scaled_control_masks;
  if (max_output_width > 0 && control_masks.canvas_width() > static_cast<size_t>(max_output_width)) {
    scaled_control_masks = control_masks;
    if (!scaled_control_masks->scale_to_max_output_width(max_output_width)) {
      status_ = CudaStatus(cudaErrorInvalidValue, "max_output_width removes one or more N-image seam classes");
      return;
    }
  }
  const ControlMasksN& masks = scaled_control_masks ? *scaled_control_masks : control_masks;

  const int n = static_cast<int>(masks.img_col.size());
  stitch_context_ =
      std::make_unique<StitchingContextN<T_pipeline, T_compute>>(batch_size, /*is_hard=*/(num_levels == 0));
  stitch_context_->n_images = n;
  stitch_context_->remap_x.resize(n);
  stitch_context_->remap_y.resize(n);

  const int canvas_w = static_cast<int>(masks.canvas_width());
  const int canvas_h = static_cast<int>(masks.canvas_height());
  if (!quiet) {
    std::cout << "Stitched (N-image) canvas size: " << canvas_w << " x " << canvas_h << std::endl;
  }

  // Canvas manager with N positions
  std::vector<cv::Point> positions;
  positions.reserve(n);
  for (int i = 0; i < n; ++i)
    positions.emplace_back(masks.positions[i].xpos, masks.positions[i].ypos);
  canvas_manager_ = std::make_unique<CanvasManagerN>(
      CanvasInfo{.width = canvas_w, .height = canvas_h, .positions = positions},
      /*minimize_blend=*/minimize_blend_,
      /*overlap_pad=*/blend_roi::scaled_overlap_padding(control_masks.canvas_width(), canvas_w));
  for (int i = 0; i < n; ++i)
    canvas_manager_->set_remap_size(i, masks.img_col[i].size());

  cv::Mat seam_index_padded = canvas_manager_->convertMaskMat(masks.whole_seam_mask_indexed);
  assert(!seam_index_padded.empty());
  seam_index_padded = seam_index_padded.clone();

  // Seam mask setup
  if (stitch_context_->is_hard_seam()) {
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_index_padded);
  } else {
    // Soft seam: Build an N-channel one-hot seam mask as base scalars [H x W x N] (optionally cropped).
    cv::Mat seam_index_for_blend = seam_index_padded;
    if (minimize_blend_) {
      stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_index_padded);

      const blend_roi::Regions regions =
          blend_roi::select_regions(seam_index_padded, num_levels, canvas_manager_->overlap_padding());
      write_roi_canvas_ = regions.write;
      blend_roi_canvas_ = regions.blend;
      if (blend_roi_canvas_.area() > 0 &&
          !blend_roi::hard_baseline_covers_soft_owners_outside_write(
              seam_index_padded, positions, masks.img_col, masks.img_row, write_roi_canvas_)) {
        write_roi_canvas_ = {};
        blend_roi_canvas_ = {};
      }
      if (blend_roi_canvas_.width > 0 && blend_roi_canvas_.height > 0) {
        seam_index_for_blend = seam_index_padded(blend_roi_canvas_);
      } else {
        // No seam boundaries detected; leave blend/write ROIs empty so downstream processing follows
        // its standard soft-seam remap+blend behavior rather than a special fast path.
        blend_roi_canvas_ = {};
        write_roi_canvas_ = {};
        remap_rois_.assign(n, {});
      }

      if (blend_roi_canvas_.width > 0 && blend_roi_canvas_.height > 0) {
        remap_rois_.resize(n);
        for (int i = 0; i < n; ++i) {
          remap_rois_[i] =
              blend_roi::remap_roi(canvas_manager_->canvas_positions()[i], masks.img_col[i].size(), blend_roi_canvas_);
        }
      }
    }

    cv::Mat seam_color_u8 = ControlMasksN::split_to_channels(seam_index_for_blend, n);
    cv::Mat seam_color_f;
    seam_color_u8.convertTo(seam_color_f, CV_MAKETYPE(CV_32F, n));

    BaseScalar_t<T_compute>* d_mask = nullptr;
    const size_t mask_bytes = seam_color_f.total() * seam_color_f.elemSize();
    auto cuerr = cudaMalloc(reinterpret_cast<void**>(&d_mask), mask_bytes);
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }
    stitch_context_->cudaBlendSoftSeam.reset(d_mask);
    cuerr = cudaMemcpy(d_mask, seam_color_f.data, mask_bytes, cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }

    const int blend_w = (minimize_blend_ && blend_roi_canvas_.width > 0) ? blend_roi_canvas_.width : canvas_w;
    const int blend_h = (minimize_blend_ && blend_roi_canvas_.height > 0) ? blend_roi_canvas_.height : canvas_h;

    stitch_context_->cudaFull.resize(n);
    stitch_context_->cudaFull_raw.resize(n);
    for (int i = 0; i < n; ++i) {
      stitch_context_->cudaFull[i] = std::make_unique<CudaMat<T_compute>>(batch_size, blend_w, blend_h);
      stitch_context_->cudaFull_raw[i] = stitch_context_->cudaFull[i]->data_raw();
    }
    stitch_context_->cudaBlendOut = std::make_unique<CudaMat<T_compute>>(batch_size, blend_w, blend_h);

    // Create the blending context for this N once; buffers are allocated lazily on first blend.
    switch (n) {
      case 2:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 2>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 3:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 3>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 4:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 4>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 5:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 5>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 6:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 6>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 7:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 7>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      case 8:
        stitch_context_->laplacian_blend_context
            .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 8>>(
                blend_w, blend_h, num_levels, batch_size);
        break;
      default:
        status_ = CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (supported 2..8)");
        return;
    }
  }

  // Load remappers to device
  for (int i = 0; i < n; ++i) {
    stitch_context_->remap_x[i] = std::make_unique<CudaMat<uint16_t>>(masks.img_col[i]);
    stitch_context_->remap_y[i] = std::make_unique<CudaMat<uint16_t>>(masks.img_row[i]);
  }

  // Device-resident metadata for the fused hard-seam kernel (used when num_levels == 0).
  {
    std::vector<const uint16_t*> h_remap_x_ptrs(n, nullptr);
    std::vector<const uint16_t*> h_remap_y_ptrs(n, nullptr);
    std::vector<int2> h_offsets(n);
    std::vector<int2> h_sizes(n);
    for (int i = 0; i < n; ++i) {
      h_remap_x_ptrs[i] = stitch_context_->remap_x[i]->data();
      h_remap_y_ptrs[i] = stitch_context_->remap_y[i]->data();
      const auto& pos = canvas_manager_->canvas_positions()[i];
      h_offsets[i] = int2{pos.x, pos.y};
      h_sizes[i] = int2{stitch_context_->remap_x[i]->width(), stitch_context_->remap_x[i]->height()};
    }

    CudaSurface<T_pipeline>* d_inputs = nullptr;
    auto cuerr =
        cudaMalloc(reinterpret_cast<void**>(&d_inputs), static_cast<size_t>(n) * sizeof(CudaSurface<T_pipeline>));
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }
    stitch_context_->d_input_surfaces.reset(d_inputs);

    const uint16_t** d_remap_x = nullptr;
    cuerr = cudaMalloc(reinterpret_cast<void**>(&d_remap_x), static_cast<size_t>(n) * sizeof(uint16_t*));
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }
    stitch_context_->d_remap_x_ptrs.reset(d_remap_x);
    cuerr = cudaMemcpy(
        stitch_context_->d_remap_x_ptrs.get(),
        h_remap_x_ptrs.data(),
        static_cast<size_t>(n) * sizeof(uint16_t*),
        cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }

    const uint16_t** d_remap_y = nullptr;
    cuerr = cudaMalloc(reinterpret_cast<void**>(&d_remap_y), static_cast<size_t>(n) * sizeof(uint16_t*));
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }
    stitch_context_->d_remap_y_ptrs.reset(d_remap_y);
    cuerr = cudaMemcpy(
        stitch_context_->d_remap_y_ptrs.get(),
        h_remap_y_ptrs.data(),
        static_cast<size_t>(n) * sizeof(uint16_t*),
        cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }

    int2* d_offsets = nullptr;
    cuerr = cudaMalloc(reinterpret_cast<void**>(&d_offsets), static_cast<size_t>(n) * sizeof(int2));
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }
    stitch_context_->d_offsets.reset(d_offsets);
    cuerr = cudaMemcpy(
        stitch_context_->d_offsets.get(),
        h_offsets.data(),
        static_cast<size_t>(n) * sizeof(int2),
        cudaMemcpyHostToDevice);
    if (cuerr != cudaSuccess) {
      status_ = CudaStatus(cuerr);
      return;
    }

    if (stitch_context_->is_hard_seam() || minimize_blend_) {
      int2* d_sizes = nullptr;
      cuerr = cudaMalloc(reinterpret_cast<void**>(&d_sizes), static_cast<size_t>(n) * sizeof(int2));
      if (cuerr != cudaSuccess) {
        status_ = CudaStatus(cuerr);
        return;
      }
      stitch_context_->d_remap_sizes.reset(d_sizes);
      cuerr = cudaMemcpy(
          stitch_context_->d_remap_sizes.get(),
          h_sizes.data(),
          static_cast<size_t>(n) * sizeof(int2),
          cudaMemcpyHostToDevice);
      if (cuerr != cudaSuccess) {
        status_ = CudaStatus(cuerr);
        return;
      }
    }
  }
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::remap_soft(
    const CudaMat<T_pipeline>& input,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    CudaMat<T_compute>& dest_canvas,
    int dest_x,
    int dest_y,
    int batch_size,
    cudaStream_t stream) {
  const T_pipeline default_pixel = T_pipeline{};
  return batched_remap_kernel_ex_offset(
      input.surface(),
      dest_canvas.surface(),
      map_x.data(),
      map_y.data(),
      default_pixel,
      batch_size,
      map_x.width(),
      map_y.height(),
      dest_x,
      dest_y,
      /*no_unmapped_write=*/false,
      stream);
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::remap_hard(
    const CudaMat<T_pipeline>& input,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    uint8_t image_index,
    const CudaMat<unsigned char>& dest_index_map,
    CudaMat<T_pipeline>& dest_canvas,
    int dest_x,
    int dest_y,
    int batch_size,
    cudaStream_t stream) {
  const T_pipeline default_pixel = T_pipeline{};
  return batched_remap_kernel_ex_offset_with_dest_map(
      input.surface(),
      dest_canvas.surface(),
      map_x.data(),
      map_y.data(),
      default_pixel,
      image_index,
      dest_index_map.data(),
      batch_size,
      map_x.width(),
      map_y.height(),
      dest_x,
      dest_y,
      stream);
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::blend_soft_dispatch(
    const std::vector<const BaseScalar_t<T_compute>*>& d_ptrs,
    cudaStream_t stream) {
  const int n = stitch_context_->n_images;
  const int C = detailN::num_channels_v<T_compute>;
  auto d_mask = stitch_context_->cudaBlendSoftSeam.get();
  auto out = stitch_context_->cudaBlendOut->data_raw();

#define BLEND_N_CASE(NVAL, CH)                                                            \
  do {                                                                                    \
    auto& ctx = std::get<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, NVAL>>( \
        stitch_context_->laplacian_blend_context);                                        \
    return CudaStatus(                                                                    \
        cudaBatchedLaplacianBlendWithContextN<BaseScalar_t<T_compute>, float, NVAL, CH>(  \
            d_ptrs, d_mask, out, ctx, stream));                                           \
  } while (0)

  if (C == 3) {
    switch (n) {
      case 2:
        BLEND_N_CASE(2, 3);
      case 3:
        BLEND_N_CASE(3, 3);
      case 4:
        BLEND_N_CASE(4, 3);
      case 5:
        BLEND_N_CASE(5, 3);
      case 6:
        BLEND_N_CASE(6, 3);
      case 7:
        BLEND_N_CASE(7, 3);
      case 8:
        BLEND_N_CASE(8, 3);
      default:
        return CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (3ch): supported 2..8");
    }
  }
  if (C == 4) {
    switch (n) {
      case 2:
        BLEND_N_CASE(2, 4);
      case 3:
        BLEND_N_CASE(3, 4);
      case 4:
        BLEND_N_CASE(4, 4);
      case 5:
        BLEND_N_CASE(5, 4);
      case 6:
        BLEND_N_CASE(6, 4);
      case 7:
        BLEND_N_CASE(7, 4);
      case 8:
        BLEND_N_CASE(8, 4);
      default:
        return CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (4ch): supported 2..8");
    }
  }

  return CudaStatus(cudaErrorInvalidValue, "Unsupported pixel channels (expect 3 or 4)");
#undef BLEND_N_CASE
}

template <typename T_pipeline, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPanoN<T_pipeline, T_compute>::process(
    const std::vector<const CudaMat<T_pipeline>*>& inputs,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  CUDA_RETURN_IF_ERROR(status_);
  if (!canvas)
    return CudaStatus(cudaErrorInvalidDevicePointer, "Canvas must be provided");
  if ((int)inputs.size() != stitch_context_->n_images)
    return CudaStatus(cudaErrorInvalidValue, "inputs size != N");
  for (auto* in : inputs) {
    if (!in || in->batch_size() != stitch_context_->batch_size())
      return CudaStatus(cudaErrorInvalidValue, "Mismatched batch sizes");
  }
  if (canvas->batch_size() != stitch_context_->batch_size())
    return CudaStatus(cudaErrorInvalidValue, "Canvas batch mismatch");

  const T_pipeline default_pixel = T_pipeline{};

  if (stitch_context_->is_hard_seam()) {
    auto cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    std::vector<CudaSurface<T_pipeline>> h_inputs(stitch_context_->n_images);
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      h_inputs[i] = inputs[i]->surface();
    }
    cuerr = cudaMemcpyAsync(
        stitch_context_->d_input_surfaces.get(),
        h_inputs.data(),
        static_cast<size_t>(stitch_context_->n_images) * sizeof(CudaSurface<T_pipeline>),
        cudaMemcpyHostToDevice,
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    cuerr = batched_remap_hard_seam_kernel_n<T_pipeline>(
        stitch_context_->d_input_surfaces.get(),
        stitch_context_->d_remap_x_ptrs.get(),
        stitch_context_->d_remap_y_ptrs.get(),
        stitch_context_->d_offsets.get(),
        stitch_context_->d_remap_sizes.get(),
        stitch_context_->n_images,
        stitch_context_->cudaBlendHardSeam->data(),
        canvas->surface(),
        stitch_context_->batch_size(),
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    return std::move(canvas);
  }

  const bool use_minimized_blend = minimize_blend_ && blend_roi_canvas_.width > 0 && blend_roi_canvas_.height > 0 &&
      write_roi_canvas_.width > 0 && write_roi_canvas_.height > 0;
  if (use_minimized_blend) {
    if (!stitch_context_->cudaBlendHardSeam) {
      return CudaStatus(cudaErrorInvalidValue, "minimize_blend requested but hard seam mask was not initialized");
    }

    auto cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);
    // The minimized soft-seam path remaps only intersections into these reusable buffers.
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      cuerr = cudaMemsetAsync(stitch_context_->cudaFull[i]->data(), 0, stitch_context_->cudaFull[i]->size(), stream);
      if (cuerr != cudaSuccess)
        return CudaStatus(cuerr);
    }

    // First fill the entire canvas using a fused hard-seam remap. This gives a correct baseline outside the
    // soft seam ROI at much lower cost than remapping N full-frame buffers.
    std::vector<CudaSurface<T_pipeline>> h_inputs(stitch_context_->n_images);
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      h_inputs[i] = inputs[i]->surface();
    }
    cuerr = cudaMemcpyAsync(
        stitch_context_->d_input_surfaces.get(),
        h_inputs.data(),
        static_cast<size_t>(stitch_context_->n_images) * sizeof(CudaSurface<T_pipeline>),
        cudaMemcpyHostToDevice,
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    cuerr = batched_remap_hard_seam_kernel_n<T_pipeline>(
        stitch_context_->d_input_surfaces.get(),
        stitch_context_->d_remap_x_ptrs.get(),
        stitch_context_->d_remap_y_ptrs.get(),
        stitch_context_->d_offsets.get(),
        stitch_context_->d_remap_sizes.get(),
        stitch_context_->n_images,
        stitch_context_->cudaBlendHardSeam->data(),
        canvas->surface(),
        stitch_context_->batch_size(),
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    // Remap only each image's intersection with the blend ROI into the compute buffers.
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      const auto& mx = *stitch_context_->remap_x[i];
      const auto& my = *stitch_context_->remap_y[i];
      assert(static_cast<int>(remap_rois_.size()) == stitch_context_->n_images);
      const blend_roi::RemapRoi& ri = remap_rois_[i];

      cuerr = batched_remap_kernel_ex_offset_roi(
          inputs[i]->surface(),
          stitch_context_->cudaFull[i]->surface(),
          mx.data(),
          my.data(),
          default_pixel,
          stitch_context_->batch_size(),
          mx.width(),
          mx.height(),
          ri.offset_x,
          ri.offset_y,
          ri.roi.x,
          ri.roi.y,
          ri.roi.width,
          ri.roi.height,
          /*no_unmapped_write=*/false,
          stream);
      if (cuerr != cudaSuccess)
        return CudaStatus(cuerr);
    }

    {
      CudaStatus s = blend_soft_dispatch(stitch_context_->cudaFull_raw, stream);
      if (!s.ok())
        return s;
    }

    {
      const int src_x = write_roi_canvas_.x - blend_roi_canvas_.x;
      const int src_y = write_roi_canvas_.y - blend_roi_canvas_.y;
      cuerr = copy_roi_batched<T_compute, T_pipeline>(
          stitch_context_->cudaBlendOut->surface(),
          /*regionWidth=*/write_roi_canvas_.width,
          /*regionHeight=*/write_roi_canvas_.height,
          /*srcROI_x=*/src_x,
          /*srcROI_y=*/src_y,
          canvas->surface(),
          /*offsetX=*/write_roi_canvas_.x,
          /*offsetY=*/write_roi_canvas_.y,
          stitch_context_->batch_size(),
          stream);
      if (cuerr != cudaSuccess)
        return CudaStatus(cuerr);
    }

    return std::move(canvas);
  }

  // Soft seam: remap each input into its own full canvas buffer, then N-way blend.
  // Remap only touches each input's projected footprint, so clear the reusable buffers first.
  for (int i = 0; i < stitch_context_->n_images; ++i) {
    auto cuerr = cudaMemsetAsync(stitch_context_->cudaFull[i]->data(), 0, stitch_context_->cudaFull[i]->size(), stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);
  }
  for (int i = 0; i < stitch_context_->n_images; ++i) {
    const auto& mx = *stitch_context_->remap_x[i];
    const auto& my = *stitch_context_->remap_y[i];
    int dx = canvas_manager_->canvas_positions()[i].x;
    int dy = canvas_manager_->canvas_positions()[i].y;
    CudaStatus s =
        remap_soft(*inputs[i], mx, my, *stitch_context_->cudaFull[i], dx, dy, stitch_context_->batch_size(), stream);
    if (!s.ok())
      return s;
  }

  {
    CudaStatus s = blend_soft_dispatch(stitch_context_->cudaFull_raw, stream);
    if (!s.ok())
      return s;
  }

  {
    auto cuerr = copy_roi_batched<T_compute, T_pipeline>(
        stitch_context_->cudaBlendOut->surface(),
        /*regionWidth=*/stitch_context_->cudaBlendOut->width(),
        /*regionHeight=*/stitch_context_->cudaBlendOut->height(),
        /*srcROI_x=*/0,
        /*srcROI_y=*/0,
        canvas->surface(),
        /*offsetX=*/0,
        /*offsetY=*/0,
        stitch_context_->batch_size(),
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);
  }

  return std::move(canvas);
}

} // namespace cuda
} // namespace pano
} // namespace hm
