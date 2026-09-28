#pragma once

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <optional>
#include <type_traits>

#include "cupano/cuda/cudaMakeFull.h"
#include "cupano/pano/blendRoi.h"
#include "cupano/pano/featherMask.h"

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
    BlendSettings blend,
    const ControlMasksN& control_masks,
    bool minimize_blend,
    bool quiet,
    int max_output_width,
    bool compact_workspace)
    : blend_(blend), compact_workspace_(compact_workspace), minimize_blend_(minimize_blend && blend.is_soft()) {
  if (const std::string invalid = blend.Validate(); !invalid.empty()) {
    status_ = CudaStatus(cudaErrorInvalidValue, invalid);
    return;
  }
  if (blend.mode == BlendMode::kAlpha && std::is_integral_v<BaseScalar_t<T_compute>>) {
    status_ =
        CudaStatus(cudaErrorNotSupported, "Alpha blending requires floating-point compute to preserve feather weights");
    return;
  }
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
      std::make_unique<StitchingContextN<T_pipeline, T_compute>>(batch_size, /*is_hard=*/blend.is_hard_seam());
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
  cv::Mat seam_index_for_roi = seam_index_padded;

  // Seam mask setup
  if (stitch_context_->is_hard_seam()) {
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_index_padded);
  } else {
    // Soft seam mask as base scalars [H x W x N], optionally cropped to the blend ROI.
    //
    // Alpha mode resolves its weights here, on the full padded canvas. Distance transforms are
    // non-local, so building them from a cropped seam would invent a boundary at the crop edge;
    // the field is cropped afterwards instead. Resolving first also yields the radius the ROI
    // padding below has to cover.
    cv::Mat feather_weights_full;
    float feather_roi_radius_px = 0.0f;
    if (blend.mode == BlendMode::kAlpha) {
      feather::Params feather_params;
      feather_params.fraction = blend.feather_fraction;
      feather::Result feathered =
          feather::build_weights(seam_index_padded, masks.img_col, masks.img_row, positions, n, feather_params);
      if (!feathered.error.empty()) {
        status_ = CudaStatus(cudaErrorInvalidValue, feathered.error);
        return;
      }
      feather_weights_full = std::move(feathered.weights);
      feather_radius_px_ = feathered.radius_px;
      feather_roi_radius_px = feathered.requested_radius_px;
      // A coverage hole inside an overlap moves a seam, so the ROI has to be derived from the
      // labels the field was actually built from, not from the ones handed in.
      if (!feathered.corrected_labels.empty())
        seam_index_for_roi = feathered.corrected_labels;
      if (!quiet && feathered.overlap_capped) {
        std::cout << "Alpha blend feather capped on " << (100.0f * feathered.capped_seam_fraction)
                  << "% of the seam (tightest " << feathered.min_seam_radius_px << " px, widest " << feather_radius_px_
                  << " px, requested " << feathered.requested_radius_px
                  << " px) to stay inside the contributing cameras' coverage" << std::endl;
      }
    }

    cv::Mat seam_index_for_blend = seam_index_padded;
    if (minimize_blend_) {
      stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(seam_index_padded);

      // The crossfade spans radius/2 either side of the seam, so the write ROI must cover it.
      const int feather_pad =
          blend.mode == BlendMode::kAlpha ? static_cast<int>(std::ceil(feather_roi_radius_px / 2.0f)) + 1 : 0;
      // Sized from the corrected labels, which is independent of the guard below rather than
      // subsumed by it: the guard only inspects pixels *outside* the ROI, so a coverage hole that
      // starts just inside it passes, and the seam the correction moves to that hole's edge then
      // feathers outside a ROI derived from the labels handed in.
      const blend_roi::Regions regions = blend_roi::select_regions(
          seam_index_for_roi, blend.roi_levels(), std::max(canvas_manager_->overlap_padding(), feather_pad));
      write_roi_canvas_ = regions.write;
      blend_roi_canvas_ = regions.blend;
      // Deliberately the original labels, not the corrected ones the field was built from: this
      // guards the hard baseline in cudaBlendHardSeam, which is built from the originals. A
      // corrected owner covers by construction, so checking those would only ever fail where no
      // camera covers at all, which disables the guard.
      if (blend_roi_canvas_.area() > 0 &&
          !blend_roi::hard_baseline_covers_soft_owners_outside_write(
              seam_index_padded, positions, masks.img_col, masks.img_row, write_roi_canvas_)) {
        write_roi_canvas_ = {};
        blend_roi_canvas_ = {};
      }
      if (blend_roi_canvas_.width > 0 && blend_roi_canvas_.height > 0) {
        seam_index_for_blend = seam_index_padded(blend_roi_canvas_);
        if (!feather_weights_full.empty()) {
          feather_weights_full = feather_weights_full(blend_roi_canvas_).clone();
        }
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

    // Either a one-hot CV_8U partition or the CV_32F feathered field; both convert below.
    cv::Mat seam_weights =
        feather_weights_full.empty() ? ControlMasksN::split_to_channels(seam_index_for_blend, n) : feather_weights_full;
    cv::Mat seam_color_f;
#if GPU_HAS_BF16
    // Unconditional despite sitting in the soft-seam branch: a static_assert fires whenever the
    // constructor is instantiated, so this also rejects bf16 hard-seam stitchers, which would
    // otherwise be fine. Nothing instantiates bf16 today.
    // Every other scalar has its own OpenCV depth, but cudaPixelTypeToCvType maps bf16 onto the
    // CV_16F codes (cudaMat.cpp), so a bf16 mask would be written as IEEE half and read as bf16.
    static_assert(
        !std::is_same_v<BaseScalar_t<T_compute>, gpu_bfloat16>,
        "bfloat16 has no distinct OpenCV depth; the soft-seam mask would be written as IEEE half");
#endif
    // The blend kernels read this buffer as BaseScalar_t<T_compute>, so the mask must carry that
    // scalar depth. Converting to a fixed CV_32F made half pipelines reinterpret float bit patterns
    // as pairs of __half. Mirrors cudaPano.inl and cudaPano3.inl; convertTo takes only the depth
    // from rtype and keeps the source's channel count.
    seam_weights.convertTo(seam_color_f, cudaPixelTypeToCvType(CudaTypeToPixelType<BaseScalar_t<T_compute>>::value));

    const int blend_w = (minimize_blend_ && blend_roi_canvas_.width > 0) ? blend_roi_canvas_.width : canvas_w;
    const int blend_h = (minimize_blend_ && blend_roi_canvas_.height > 0) ? blend_roi_canvas_.height : canvas_h;

    BaseScalar_t<T_compute>* d_mask = nullptr;
    const size_t mask_bytes = seam_color_f.total() * seam_color_f.elemSize();
    // Checked rather than asserted, because asserts are compiled out of optimized builds and each
    // of these mismatches corrupts the blend silently rather than faulting.
    //
    // The dimensions matter most. BatchedBlendKernelN indexes the mask with the blend buffers'
    // stride, so a seam mask larger than the canvas - which ControlMasksN never validates and
    // CanvasManagerN::convertMaskMat only pads, never crops - reads the right number of bytes at
    // the wrong stride and silently picks the wrong contributor.
    if (seam_color_f.cols != blend_w || seam_color_f.rows != blend_h || seam_color_f.channels() != n ||
        seam_color_f.elemSize1() != sizeof(BaseScalar_t<T_compute>)) {
      status_ = CudaStatus(
          cudaErrorInvalidValue,
          "Soft-seam mask shape or scalar type does not match the blend buffers (expected " + std::to_string(blend_w) +
              "x" + std::to_string(blend_h) + "x" + std::to_string(n) + " of " +
              std::to_string(sizeof(BaseScalar_t<T_compute>)) + "-byte scalars, got " +
              std::to_string(seam_color_f.cols) + "x" + std::to_string(seam_color_f.rows) + "x" +
              std::to_string(seam_color_f.channels()) + " of " + std::to_string(seam_color_f.elemSize1()) + ")");
      return;
    }
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

    stitch_context_->cudaFull.resize(n);
    stitch_context_->cudaFull_raw.resize(n);
    for (int i = 0; i < n; ++i) {
      stitch_context_->cudaFull[i] = std::make_unique<CudaMat<T_compute>>(batch_size, blend_w, blend_h);
      stitch_context_->cudaFull_raw[i] = stitch_context_->cudaFull[i]->data_raw();
    }
    if (compact_workspace) {
      stitch_context_->cudaBlendOut =
          std::make_unique<CudaMat<T_compute>>(stitch_context_->cudaFull[0]->data(), batch_size, blend_w, blend_h);
    } else {
      stitch_context_->cudaBlendOut = std::make_unique<CudaMat<T_compute>>(batch_size, blend_w, blend_h);
    }

    if (blend.mode == BlendMode::kAlpha) {
      if (n < 2 || n > 8) {
        status_ = CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (supported 2..8)");
        return;
      }
      // Alpha mode is a single pass over level 0, so it needs no pyramid context at all - only a
      // device array of the (fixed) remap destinations. Uploading it here keeps the per-frame blend
      // free of host-to-device traffic.
      const BaseScalar_t<T_compute>** d_inputs = nullptr;
      const size_t inputs_bytes = static_cast<size_t>(n) * sizeof(const BaseScalar_t<T_compute>*);
      auto inputs_err = cudaMalloc(reinterpret_cast<void**>(&d_inputs), inputs_bytes);
      if (inputs_err != cudaSuccess) {
        status_ = CudaStatus(inputs_err);
        return;
      }
      stitch_context_->d_blend_inputs.reset(d_inputs);
      inputs_err = cudaMemcpy(d_inputs, stitch_context_->cudaFull_raw.data(), inputs_bytes, cudaMemcpyHostToDevice);
      if (inputs_err != cudaSuccess) {
        status_ = CudaStatus(inputs_err);
        return;
      }
    } else {
      // Create the blending context for this N once; buffers are allocated lazily on first blend.
      switch (n) {
        case 2:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 2>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 3:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 3>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 4:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 4>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 5:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 5>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 6:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 6>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 7:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 7>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        case 8:
          stitch_context_->laplacian_blend_context
              .template emplace<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 8>>(
                  blend_w, blend_h, blend.num_levels, batch_size, compact_workspace);
          break;
        default:
          status_ = CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (supported 2..8)");
          return;
      }
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
template <typename T_input>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::remap_soft(
    const CudaMat<T_input>& input,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    CudaMat<T_compute>& dest_canvas,
    int dest_x,
    int dest_y,
    int batch_size,
    cudaStream_t stream) {
  const T_input default_pixel = T_input{};
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
template <typename T_input>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::remap_hard(
    const CudaMat<T_input>& input,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    uint8_t image_index,
    const CudaMat<unsigned char>& dest_index_map,
    CudaMat<T_pipeline>& dest_canvas,
    int dest_x,
    int dest_y,
    int batch_size,
    cudaStream_t stream) {
  const T_input default_pixel = T_input{};
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
  const int blend_width = stitch_context_->cudaBlendOut->width();
  const int blend_height = stitch_context_->cudaBlendOut->height();
  // Alpha mode reads its inputs through the device pointer table uploaded at construction, which
  // mirrors cudaFull_raw. Both call sites pass exactly that vector.
  assert(blend_.mode != BlendMode::kAlpha || d_ptrs.data() == stitch_context_->cudaFull_raw.data());

#define BLEND_N_CASE(NVAL, CH)                                                            \
  do {                                                                                    \
    if (blend_.mode == BlendMode::kAlpha) {                                               \
      return CudaStatus(                                                                  \
          cudaBatchedAlphaBlendN<BaseScalar_t<T_compute>, float, NVAL, CH>(               \
              stitch_context_->d_blend_inputs.get(),                                      \
              d_mask,                                                                     \
              out,                                                                        \
              blend_width,                                                                \
              blend_height,                                                               \
              stitch_context_->batch_size(),                                              \
              stream));                                                                   \
    }                                                                                     \
    auto& ctx = std::get<CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, NVAL>>( \
        stitch_context_->laplacian_blend_context);                                        \
    return CudaStatus(                                                                    \
        cudaBatchedLaplacianBlendWithContextN<BaseScalar_t<T_compute>, float, NVAL, CH>(  \
            d_ptrs, d_mask, out, ctx, stream, /*cacheMaskPyramid=*/true));                \
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
template <typename T_input>
CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> CudaStitchPanoN<T_pipeline, T_compute>::process(
    const std::vector<const CudaMat<T_input>*>& inputs,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T_pipeline>>&& canvas) {
  static_assert(
      std::is_same_v<T_input, T_pipeline> ||
          (std::is_same_v<T_input, Rgb10A2> && std::is_same_v<T_pipeline, half4> && std::is_same_v<T_compute, half4>),
      "Packed RGB10A2 inputs require half4 pipeline and compute types");
  CUDA_RETURN_IF_ERROR(status_);
  if (!canvas) {
    if constexpr (std::is_same_v<T_pipeline, T_compute>) {
      if (compact_workspace_ && !stitch_context_->is_hard_seam() && !minimizes_blend()) {
        canvas = std::make_unique<CudaMat<T_pipeline>>(
            stitch_context_->cudaBlendOut->data(), batch_size(), canvas_width(), canvas_height());
      }
    }
    if (!canvas)
      canvas = std::make_unique<CudaMat<T_pipeline>>(batch_size(), canvas_width(), canvas_height());
    if (!canvas->is_valid())
      return CudaStatus(cudaErrorMemoryAllocation, "Could not allocate panorama output");
  }
  if ((int)inputs.size() != stitch_context_->n_images)
    return CudaStatus(cudaErrorInvalidValue, "inputs size != N");
  for (auto* in : inputs) {
    if (!in || in->batch_size() != stitch_context_->batch_size())
      return CudaStatus(cudaErrorInvalidValue, "Mismatched batch sizes");
  }
  if (canvas->batch_size() != stitch_context_->batch_size())
    return CudaStatus(cudaErrorInvalidValue, "Canvas batch mismatch");

  const T_input default_pixel = T_input{};

  if (stitch_context_->is_hard_seam()) {
    auto cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    std::vector<CudaSurface<T_input>> h_inputs(stitch_context_->n_images);
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      h_inputs[i] = inputs[i]->surface();
    }
    cuerr = cudaMemcpyAsync(
        reinterpret_cast<CudaSurface<T_input>*>(stitch_context_->d_input_surfaces.get()),
        h_inputs.data(),
        static_cast<size_t>(stitch_context_->n_images) * sizeof(CudaSurface<T_input>),
        cudaMemcpyHostToDevice,
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    cuerr = batched_remap_hard_seam_kernel_n<T_input, T_pipeline>(
        reinterpret_cast<CudaSurface<T_input>*>(stitch_context_->d_input_surfaces.get()),
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
    std::vector<CudaSurface<T_input>> h_inputs(stitch_context_->n_images);
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      h_inputs[i] = inputs[i]->surface();
    }
    cuerr = cudaMemcpyAsync(
        reinterpret_cast<CudaSurface<T_input>*>(stitch_context_->d_input_surfaces.get()),
        h_inputs.data(),
        static_cast<size_t>(stitch_context_->n_images) * sizeof(CudaSurface<T_input>),
        cudaMemcpyHostToDevice,
        stream);
    if (cuerr != cudaSuccess)
      return CudaStatus(cuerr);

    cuerr = batched_remap_hard_seam_kernel_n<T_input, T_pipeline>(
        reinterpret_cast<CudaSurface<T_input>*>(stitch_context_->d_input_surfaces.get()),
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

  if constexpr (std::is_same_v<T_pipeline, T_compute>) {
    if (canvas->data_raw() == stitch_context_->cudaBlendOut->data_raw())
      return std::move(canvas);
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
