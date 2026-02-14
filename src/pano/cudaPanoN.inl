#pragma once

#include <opencv2/imgproc.hpp>

namespace hm {
namespace pano {
namespace cuda {

namespace detailN {
constexpr inline float3 neg(const float3& f) { return float3{.x = -f.x, .y = -f.y, .z = -f.z}; }

template <typename T>
inline constexpr int num_channels_v = sizeof(T) / sizeof(BaseScalar_t<T>);

} // namespace detailN

template <typename T_pipeline, typename T_compute>
CudaStitchPanoN<T_pipeline, T_compute>::CudaStitchPanoN(
    int batch_size,
    int num_levels,
    const ControlMasksN& control_masks,
    bool match_exposure,
    bool quiet)
    : match_exposure_(match_exposure) {
  if (!control_masks.is_valid()) {
    status_ = CudaStatus(cudaError_t::cudaErrorFileNotFound, "Stitching masks (N-image) could not be loaded");
    return;
  }

  const int n = static_cast<int>(control_masks.img_col.size());
  stitch_context_ = std::make_unique<StitchingContextN<T_pipeline, T_compute>>(batch_size, /*is_hard=*/(num_levels == 0));
  stitch_context_->n_images = n;
  stitch_context_->remap_x.resize(n);
  stitch_context_->remap_y.resize(n);

  const int canvas_w = static_cast<int>(control_masks.canvas_width());
  const int canvas_h = static_cast<int>(control_masks.canvas_height());
  if (!quiet) {
    std::cout << "Stitched (N-image) canvas size: " << canvas_w << " x " << canvas_h << std::endl;
  }

  // Canvas manager with N positions
  std::vector<cv::Point> positions;
  positions.reserve(n);
  for (int i = 0; i < n; ++i) positions.emplace_back(control_masks.positions[i].xpos, control_masks.positions[i].ypos);
  canvas_manager_ = std::make_unique<CanvasManagerN>(CanvasInfo{.width = canvas_w, .height = canvas_h, .positions = positions},
                                                     /*minimize_blend=*/!stitch_context_->is_hard_seam());
  for (int i = 0; i < n; ++i) canvas_manager_->set_remap_size(i, control_masks.img_col[i].size());

  // Seam mask setup
  if (!stitch_context_->is_hard_seam()) {
    cv::Mat seam_color = ControlMasksN::split_to_channels(control_masks.whole_seam_mask_indexed, n);
    seam_color.convertTo(seam_color, cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value));
    stitch_context_->cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(seam_color);
    stitch_context_->cudaFull.resize(n);
    for (int i = 0; i < n; ++i) {
      stitch_context_->cudaFull[i] = std::make_unique<CudaMat<T_compute>>(batch_size, seam_color.cols, seam_color.rows);
    }
  } else {
    stitch_context_->cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(control_masks.whole_seam_mask_indexed);
  }

  if (match_exposure_) {
    whole_seam_mask_image_ = control_masks.whole_seam_mask_indexed;
  }

  // Load remappers to device
  for (int i = 0; i < n; ++i) {
    stitch_context_->remap_x[i] = std::make_unique<CudaMat<uint16_t>>(control_masks.img_col[i]);
    stitch_context_->remap_y[i] = std::make_unique<CudaMat<uint16_t>>(control_masks.img_row[i]);
  }
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::remap_soft(
    const CudaMat<T_pipeline>& input,
    const CudaMat<uint16_t>& map_x,
    const CudaMat<uint16_t>& map_y,
    CudaMat<T_pipeline>& dest_canvas,
    int dest_x,
    int dest_y,
    const std::optional<float3>& image_adjustment,
    int batch_size,
    cudaStream_t stream) {
  if (image_adjustment.has_value()) {
    return batched_remap_kernel_ex_offset_adjust(
        input.surface(), dest_canvas.surface(), map_x.data(), map_y.data(), {0}, batch_size, map_x.width(), map_y.height(), dest_x, dest_y, /*no_unmapped_write=*/false, detailN::neg(*image_adjustment), stream);
  } else {
    return batched_remap_kernel_ex_offset(
        input.surface(), dest_canvas.surface(), map_x.data(), map_y.data(), {0}, batch_size, map_x.width(), map_y.height(), dest_x, dest_y, /*no_unmapped_write=*/false, stream);
  }
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
    const std::optional<float3>& image_adjustment,
    int batch_size,
    cudaStream_t stream) {
  if (image_adjustment.has_value()) {
    return batched_remap_kernel_ex_offset_with_dest_map_adjust(
        input.surface(), dest_canvas.surface(), map_x.data(), map_y.data(), {0}, image_index, dest_index_map.data(), batch_size, map_x.width(), map_y.height(), dest_x, dest_y, detailN::neg(*image_adjustment), stream);
  } else {
    return batched_remap_kernel_ex_offset_with_dest_map(
        input.surface(), dest_canvas.surface(), map_x.data(), map_y.data(), {0}, image_index, dest_index_map.data(), batch_size, map_x.width(), map_y.height(), dest_x, dest_y, stream);
  }
}

template <typename T_pipeline, typename T_compute>
CudaStatus CudaStitchPanoN<T_pipeline, T_compute>::blend_soft_dispatch(std::vector<const T_compute*>& d_ptrs, cudaStream_t stream) {
  const int n = stitch_context_->n_images;
  const int C = detailN::num_channels_v<T_compute>;
  auto d_mask = stitch_context_->cudaBlendSoftSeam->data_raw();
  auto out = stitch_context_->cudaFull[0]->data_raw();
  int W = stitch_context_->cudaFull[0]->width();
  int H = stitch_context_->cudaFull[0]->height();
  int B = stitch_context_->batch_size();

  // Note: we build a temporary context per call (avoids dynamic polymorphism); N is compile-time in each case.
#define BLEND_N_CASE(NVAL, CH)                                                                                                 \
  do {                                                                                                                         \
    CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, NVAL> ctx(W, H, /*levels=*/3, B);                                 \
    /* Build scalar-pointer list from pixel-pointer list */                                                                     \
    std::vector<const BaseScalar_t<T_compute>*> scalar_ptrs;                                                                   \
    scalar_ptrs.reserve(d_ptrs.size());                                                                                        \
    for (auto* p : d_ptrs) scalar_ptrs.push_back(reinterpret_cast<const BaseScalar_t<T_compute>*>(p));                         \
    if constexpr (CH == 3) {                                                                                                   \
      return CudaStatus(cudaBatchedLaplacianBlendWithContextN<BaseScalar_t<T_compute>, float, NVAL, 3>(                        \
          scalar_ptrs, d_mask, out, ctx, stream));                                                                             \
    } else {                                                                                                                   \
      return CudaStatus(cudaBatchedLaplacianBlendWithContextN<BaseScalar_t<T_compute>, float, NVAL, 4>(                        \
          scalar_ptrs, d_mask, out, ctx, stream));                                                                             \
    }                                                                                                                          \
  } while (0)

  if (C == 3) {
    switch (n) {
      case 2: BLEND_N_CASE(2, 3);
      case 3: BLEND_N_CASE(3, 3);
      case 4: BLEND_N_CASE(4, 3);
      case 5: BLEND_N_CASE(5, 3);
      case 6: BLEND_N_CASE(6, 3);
      case 7: BLEND_N_CASE(7, 3);
      case 8: BLEND_N_CASE(8, 3);
      default: return CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (3ch): supported 2..8");
    }
  } else if (C == 4) {
    switch (n) {
      case 2: BLEND_N_CASE(2, 4);
      case 3: BLEND_N_CASE(3, 4);
      case 4: BLEND_N_CASE(4, 4);
      case 5: BLEND_N_CASE(5, 4);
      case 6: BLEND_N_CASE(6, 4);
      case 7: BLEND_N_CASE(7, 4);
      case 8: BLEND_N_CASE(8, 4);
      default: return CudaStatus(cudaErrorInvalidValue, "Unsupported N for blend (4ch): supported 2..8");
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
  if (!canvas) return CudaStatus(cudaErrorInvalidDevicePointer, "Canvas must be provided");
  if ((int)inputs.size() != stitch_context_->n_images) return CudaStatus(cudaErrorInvalidValue, "inputs size != N");
  for (auto* in : inputs) {
    if (!in || in->batch_size() != stitch_context_->batch_size()) return CudaStatus(cudaErrorInvalidValue, "Mismatched batch sizes");
  }
  if (canvas->batch_size() != stitch_context_->batch_size()) return CudaStatus(cudaErrorInvalidValue, "Canvas batch mismatch");

  // Clear alpha if 4-channel
  if constexpr (detailN::num_channels_v<T_pipeline> == 4) {
    auto cuerr = cudaMemsetAsync(canvas->data(), 0, canvas->size(), stream);
    if (cuerr != cudaSuccess) return CudaStatus(cuerr);
  }

  // Remap all inputs to canvas
  for (int i = 0; i < stitch_context_->n_images; ++i) {
    const auto& mx = *stitch_context_->remap_x[i];
    const auto& my = *stitch_context_->remap_y[i];
    int dx = canvas_manager_->canvas_positions()[i].x;
    int dy = canvas_manager_->canvas_positions()[i].y;
    CudaStatus s = stitch_context_->is_hard_seam() ?
        remap_hard(*inputs[i], mx, my, static_cast<uint8_t>(i), *stitch_context_->cudaBlendHardSeam, *canvas, dx, dy, image_adjustment_, stitch_context_->batch_size(), stream)
                                                   :
        remap_soft(*inputs[i], mx, my, *canvas, dx, dy, image_adjustment_, stitch_context_->batch_size(), stream);
    if (!s.ok()) return s;
  }

  if (!stitch_context_->is_hard_seam()) {
    // Copy full canvas into each cudaFull[i] at their positions (simple full copy suffices since mask is N-channel over canvas)
    for (int i = 0; i < stitch_context_->n_images; ++i) {
      auto cuerr = cudaMemcpyAsync(
          stitch_context_->cudaFull[i]->data(), canvas->data(), stitch_context_->cudaFull[i]->size(), cudaMemcpyDeviceToDevice, stream);
      if (cuerr != cudaSuccess) return CudaStatus(cuerr);
    }
    // Build device pointer list
    std::vector<const T_compute*> ptrs(stitch_context_->n_images);
    for (int i = 0; i < stitch_context_->n_images; ++i) ptrs[i] = stitch_context_->cudaFull[i]->data();
    CudaStatus s = blend_soft_dispatch(ptrs, stream);
    if (!s.ok()) return s;
    // Copy blended result back to canvas
    auto cuerr = cudaMemcpyAsync(canvas->data(), stitch_context_->cudaFull[0]->data(), stitch_context_->cudaFull[0]->size(), cudaMemcpyDeviceToDevice, stream);
    if (cuerr != cudaSuccess) return CudaStatus(cuerr);
  }

  return std::move(canvas);
}

} // namespace cuda
} // namespace pano
} // namespace hm
