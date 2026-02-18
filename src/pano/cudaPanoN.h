#pragma once

#include <memory>
#include <optional>
#include <variant>
#include <vector>

#include "cupano/cuda/cudaBlendN.h"
#include "cupano/cuda/cudaRemap.h"
#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/canvasManagerN.h"
#include "cupano/pano/controlMasksN.h"
#include "cupano/pano/cudaMat.h"

namespace hm {
namespace pano {
namespace cuda {

template <typename T>
struct CudaFreeDeleter {
  void operator()(T* ptr) const noexcept {
    if (ptr)
      cudaFree(ptr);
  }
};

template <typename T_pipeline, typename T_compute>
struct StitchingContextN {
  StitchingContextN(int batch_size, bool is_hard) : batch_size_(batch_size), is_hard_seam_(is_hard) {}

  // N remap buffers (x,y for each image) allocated in constructor
  std::vector<std::unique_ptr<CudaMat<uint16_t>>> remap_x;
  std::vector<std::unique_ptr<CudaMat<uint16_t>>> remap_y;

  // Soft seam: N compute buffers and N-channel mask
  std::vector<std::unique_ptr<CudaMat<T_compute>>> cudaFull; // size N
  // Cached raw pointers to cudaFull buffers for blending (length N).
  std::vector<const BaseScalar_t<T_compute>*> cudaFull_raw;
  // Soft seam blend output (separate from inputs so cudaFull buffers never get "contaminated" by output).
  std::unique_ptr<CudaMat<T_compute>> cudaBlendOut;
  // Soft seam mask: [H x W x N] base scalars (not batched)
  std::unique_ptr<BaseScalar_t<T_compute>, CudaFreeDeleter<BaseScalar_t<T_compute>>> cudaBlendSoftSeam;

  using BlendContextVariant = std::variant<
      std::monostate,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 2>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 3>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 4>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 5>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 6>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 7>,
      CudaBatchLaplacianBlendContextN<BaseScalar_t<T_compute>, 8>>;
  BlendContextVariant laplacian_blend_context;

  // Hard seam: single-channel index map for writing
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam; // indices [0..N-1]

  // Hard seam fused-kernel metadata (device-resident, length N).
  std::unique_ptr<CudaSurface<T_pipeline>, CudaFreeDeleter<CudaSurface<T_pipeline>>> d_input_surfaces;
  std::unique_ptr<const uint16_t*, CudaFreeDeleter<const uint16_t*>> d_remap_x_ptrs;
  std::unique_ptr<const uint16_t*, CudaFreeDeleter<const uint16_t*>> d_remap_y_ptrs;
  std::unique_ptr<int2, CudaFreeDeleter<int2>> d_offsets;
  std::unique_ptr<int2, CudaFreeDeleter<int2>> d_remap_sizes;

  int n_images{0};

  int batch_size() const {
    return batch_size_;
  }
  bool is_hard_seam() const {
    return is_hard_seam_;
  }

 private:
  int batch_size_;
  bool is_hard_seam_;
};

// Generic N-image stitcher using cudaBlendN kernels; dispatches by N up to a small max.
template <typename T_pipeline, typename T_compute>
class CudaStitchPanoN {
 public:
  CudaStitchPanoN(
      int batch_size,
      int num_levels,
      const ControlMasksN& control_masks,
      bool match_exposure = false,
      bool quiet = false,
      bool minimize_blend = true);

  int canvas_width() const {
    return canvas_manager_->canvas_width();
  }
  int canvas_height() const {
    return canvas_manager_->canvas_height();
  }
  int batch_size() const {
    return stitch_context_->batch_size();
  }
  const CudaStatus status() const {
    return status_;
  }

  // Inputs are pointers to N CudaMat<T_pipeline> with same batch.
  CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process(
      const std::vector<const CudaMat<T_pipeline>*>& inputs,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 private:
  // Internal helpers
  CudaStatus remap_soft(
      const CudaMat<T_pipeline>& input,
      const CudaMat<uint16_t>& map_x,
      const CudaMat<uint16_t>& map_y,
      CudaMat<T_compute>& dest_canvas,
      int dest_x,
      int dest_y,
      const std::optional<float3>& image_adjustment,
      int batch_size,
      cudaStream_t stream);

  CudaStatus remap_hard(
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
      cudaStream_t stream);

  // Dispatch to cudaBlendN for channels=3 or 4, N in [2..8].
  CudaStatus blend_soft_dispatch(const std::vector<const BaseScalar_t<T_compute>*>& d_ptrs, cudaStream_t stream);

 private:
  struct RemapRoiInfo {
    // ROI within the remap map (local coordinates).
    cv::Rect roi;
    // Destination offsets for the full remap region in the destination surface coordinates.
    int offset_x{0};
    int offset_y{0};
  };

  std::unique_ptr<StitchingContextN<T_pipeline, T_compute>> stitch_context_;
  std::unique_ptr<CanvasManagerN> canvas_manager_;
  bool match_exposure_{false};
  bool minimize_blend_{false};
  std::optional<float3> image_adjustment_;
  std::optional<cv::Mat> whole_seam_mask_image_;
  cv::Rect blend_roi_canvas_{};
  cv::Rect write_roi_canvas_{};
  std::vector<RemapRoiInfo> remap_rois_;
  CudaStatus status_;
};

} // namespace cuda
} // namespace pano
} // namespace hm

#include "cudaPanoN.inl"
