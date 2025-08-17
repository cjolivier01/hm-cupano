#pragma once
#include <memory>
#include <optional>
#include "cupano/cuda/cudaBlend3.h"
#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/canvasManager3.h"
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/cudaMat.h"

namespace hm {
namespace pano {
namespace cuda {

/* clang-format off */
/**
 *   _____ *   *  *        *     *              *____             * 
 * |  ___| | (_)| |      | |   (_)            / ____|           | |
 * | |__ _   _ ___| |_ ___| |__  _ _ __   __ _| |     ___  _ __ | |_ _____  _| |_
 * |  __| | | / __| __/ __| '_ \| | '_ \ / _` | |    / _ \| '_ \| __/ _ \ \/ / __|
 * | |  | |_| \__ \ || (__| | | | | | | | (_| | |___| (_) | | | | ||  __/>  <| |_
 * |_|   \__,_|___/\__\___|_| |_|_|_| |_|\__, |\_____\___/|_| |_|\__\___/_/\_\\__|
 *                                        __/ |
 *                                       |___/
 *
 * OPTIMIZED VERSION with fused kernels for THREE-IMAGE blending
 */
/* clang-format on */

template <typename T_pipeline, typename T_compute>
struct StitchingContext3 {
  StitchingContext3(int batch_size, bool is_hard_seam) : batch_size_(batch_size), is_hard_seam_(is_hard_seam) {}

  // Static remap buffers for the three inputs:
  std::unique_ptr<CudaMat<uint16_t>> remap_0_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_0_y;
  std::unique_ptr<CudaMat<uint16_t>> remap_1_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_1_y;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_y;

  // Three "full"-images for blending (scratch space):
  std::unique_ptr<CudaMat<T_compute>> cudaFull0;
  std::unique_ptr<CudaMat<T_compute>> cudaFull1;
  std::unique_ptr<CudaMat<T_compute>> cudaFull2;

  // Soft-seam: 3-channel mask; or if hard seam, single-channel:
  std::unique_ptr<CudaMat<T_compute>> cudaBlendSoftSeam;
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam;

  // 3-image Laplacian-blend context (for soft-seam case):
  std::unique_ptr<CudaBatchLaplacianBlendContext3<BaseScalar_t<T_compute>>> laplacian_blend_context;

  constexpr int batch_size() const {
    return batch_size_;
  }
  constexpr bool is_hard_seam() const {
    return is_hard_seam_;
  }

 private:
  int batch_size_;
  bool is_hard_seam_;
};

template <typename T_pipeline, typename T_compute>
class CudaStitchPano3 {
 public:
  using pipeline_type = T_pipeline;
  using compute_type = T_compute;

  CudaStitchPano3(
      int batch_size,
      int num_levels,
      const ControlMasks3& control_masks,
      bool match_exposure = false,
      bool quiet = false);

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

  /**
   * @brief Process/​stitch three images into the provided canvas.
   *        inputImage0, inputImage1, inputImage2 must each have
   *        the same batch size.
   * @param inputImage0   “CudaMat” for image #0
   * @param inputImage1   “CudaMat” for image #1
   * @param inputImage2   “CudaMat” for image #2
   * @param stream        CUDA stream (0 = default)
   * @param canvas        preallocated “CudaMat” for the full panorama canvas
   *
   * First remaps each input into the canvas via their remappers, then copies
   * any “overlapping” region(s) into cudaFull0, cudaFull1, cudaFull2.  If
   * soft‐seam, calls `cudaBatchedLaplacianBlendWithContext3(...)` on those
   * three “full” images + 3‐channel mask.  If hard‐seam, it instead calls
   * a conditional “dest_map” remap.  Finally returns the updated canvas.
   */
  CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas,
      bool fused = true);

  /**
   * @brief Process/​stitch three images into the provided canvas.
   *        inputImage0, inputImage1, inputImage2 must each have
   *        the same batch size.
   * @param inputImage0   “CudaMat” for image #0
   * @param inputImage1   “CudaMat” for image #1
   * @param inputImage2   “CudaMat” for image #2
   * @param stream        CUDA stream (0 = default)
   * @param canvas        preallocated “CudaMat” for the full panorama canvas
   *
   * First remaps each input into the canvas via their remappers, then copies
   * any “overlapping” region(s) into cudaFull0, cudaFull1, cudaFull2.  If
   * soft‐seam, calls `cudaBatchedLaplacianBlendWithContext3(...)` on those
   * three “full” images + 3‐channel mask.  If hard‐seam, it instead calls
   * a conditional “dest_map” remap.  Finally returns the updated canvas.
   */
  CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process_optimized(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 protected:
  // Per-image additive adjustment (RGB) for three inputs
  struct ImageAdjust3 {
    float3 adj0;
    float3 adj1;
    float3 adj2;
  };
  // Original process_impl for compatibility
  static CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process_impl(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      StitchingContext3<T_pipeline, T_compute>& stitch_context,
      const CanvasManager3& canvas_manager,
      const std::optional<ImageAdjust3>& image_adjustment,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

  // Optimized process_impl with fused kernels
  static CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process_impl_optimized(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      StitchingContext3<T_pipeline, T_compute>& stitch_context,
      const CanvasManager3& canvas_manager,
      const std::optional<ImageAdjust3>& image_adjustment,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 private:
  // Legacy remap functions (kept for compatibility)
  static CudaStatus remap_to_surface_for_blending(
      const CudaMat<T_pipeline>& inputImage,
      const CudaMat<uint16_t>& map_x,
      const CudaMat<uint16_t>& map_y,
      CudaMat<T_pipeline>& dest_canvas,
      int dest_canvas_x,
      int dest_canvas_y,
      const std::optional<float3>& image_adjustment,
      int batch_size,
      cudaStream_t stream);

  static CudaStatus remap_to_surface_for_blending_compute(
      const CudaMat<T_pipeline>& inputImage,
      const CudaMat<uint16_t>& map_x,
      const CudaMat<uint16_t>& map_y,
      CudaMat<T_compute>& dest_canvas,
      int dest_canvas_x,
      int dest_canvas_y,
      const std::optional<float3>& image_adjustment,
      int batch_size,
      cudaStream_t stream);

  static CudaStatus remap_to_surface_for_hard_seam(
      const CudaMat<T_pipeline>& inputImage,
      const CudaMat<uint16_t>& map_x,
      const CudaMat<uint16_t>& map_y,
      uint8_t canvas_position_image_index,
      const CudaMat<unsigned char>& canvas_position_image_index_map,
      CudaMat<T_pipeline>& dest_canvas,
      int dest_canvas_x,
      int dest_canvas_y,
      const std::optional<float3>& image_adjustment,
      int batch_size,
      cudaStream_t stream);

  std::optional<ImageAdjust3> compute_image_adjustment(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2);

  cv::Mat make_n_channel_seam_image(const cv::Mat& seam_image, int n_channels);

  std::unique_ptr<StitchingContext3<T_pipeline, T_compute>> stitch_context_;
  std::unique_ptr<CanvasManager3> canvas_manager_;
  bool match_exposure_;
  std::optional<ImageAdjust3> image_adjustment_;
  std::optional<cv::Mat> whole_seam_mask_image_; // now 3‐channel if soft seam
  CudaStatus status_;
};

} // namespace cuda

// Compute additive RGB offsets for each of three images so that seams are minimized.
// Returns per-image adjustments (B,G,R) in an array indexed by image 0/1/2.
std::optional<std::array<cv::Scalar, 3>> match_seam_images3(
    const cv::Mat& image0,
    const cv::Mat& image1,
    const cv::Mat& image2,
    const cv::Mat& seam_indexed, // CV_8U, values in {0,1,2}
    int N,
    const cv::Point& topLeft0,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose = false);

} // namespace pano
} // namespace hm

#include "cudaPano3.inl"
#include "cudaPano3_optimized.inl"
