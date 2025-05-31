#pragma once

#include "cupano/cuda/cudaBlend3.h"
#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/canvasManager3.h"
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/cudaMat.h"

#include <memory>
#include <optional>

namespace hm {
namespace pano {
namespace cuda {

/**
 *   _____ _   _  _        _     _              _____             _               _
 *  / ____| | (_)| |      | |   (_)            / ____|           | |             | |
 * | (___ | |_ _ | |_  ___| |__  _ _ __   __ _| |      ___  _ __ | |_  ___ __  __| |_
 *  \___ \| __| || __|/ __| '_ \| | '_ \ / _` | |     / _ \| '_ \| __|/ _ \\ \/ /| __|
 *  ____) | |_| || |_| (__| | | | | | | | (_| | |____| (_) | | | | |_|  __/ >  < | |_
 * |_____/ \__|_| \__|\___|_| |_|_|_| |_|\__, |\_____|\___/|_| |_|\__|\___|/_/\_\ \__|
 *                                        __/ |
 *                                       |___/
 *
 * Modified for THREE‐IMAGE blending:
 *
 * Contains three remappers (for image0, image1, image2),
 * three “full”‐buffers, and uses a 3‐channel mask + 3‐image Laplacian blend.
 */
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

  // Three “full”‐images for blending (scratch space):
  std::unique_ptr<CudaMat<T_compute>> cudaFull0;
  std::unique_ptr<CudaMat<T_compute>> cudaFull1;
  std::unique_ptr<CudaMat<T_compute>> cudaFull2;

  // Soft‐seam: 3‐channel mask; or if hard seam, single‐channel:
  std::unique_ptr<CudaMat<T_compute>> cudaBlendSoftSeam; // CV type = T_compute with 3 channels
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam; // single‐channel (0/1)

  // 3‐image Laplacian‐blend context (for soft‐seam case):
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

/**
 *   _____           _        _____ _   _  _        _     _____
 *  / ____|         | |      / ____| | (_)| |      | |   |  __ \
 * | |     _   _  __| | __ _| (___ | |_ _ | |_  ___| |__ | |__) |__ _ _ __   ___
 * | |    | | | |/ _` |/ _` |\___ \| __| || __|/ __| '_ \|  ___// _` | '_ \ / _ \
 * | |____| |_| | (_| | (_| |____) | |_| || |_| (__| | | | |   | (_| | | | | (_) |
 *  \_____|\__,_|\__,_|\__,_|_____/ \__|_| \__|\___|_| |_|_|    \__,_|_| |_|\___/
 *
 * Modified to accept **three** input CudaMat images (image0, image1, image2),
 * plus a 3‐channel soft mask or single‐channel hard mask, producing a single
 * canvas output that blends all three via a 3‐image Laplacian blend.
 */
template <typename T_pipeline, typename T_compute>
class CudaStitchPano3 {
 public:
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
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 protected:
  /**
   * @brief The “real” implementation taking in three inputs, the context, and the canvas manager.
   */
  static CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process_impl(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      StitchingContext3<T_pipeline, T_compute>& stitch_context,
      const CanvasManager3& canvas_manager,
      const std::optional<float3>& image_adjustment,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 private:
  /**
   * @brief If “match_exposure_” is true, tries to compute a per‐channel offset
   * that aligns image0 vs. image1 vs. image2 across the seam.  Returns three floats.
   * Otherwise returns std::nullopt.
   */
  std::optional<float3> compute_image_adjustment(
      const CudaMat<T_pipeline>& inputImage0,
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2);

  std::unique_ptr<StitchingContext3<T_pipeline, T_compute>> stitch_context_;
  std::unique_ptr<CanvasManager3> canvas_manager_;
  bool match_exposure_;
  std::optional<float3> image_adjustment_;
  std::optional<cv::Mat> whole_seam_mask_image_; // now 3‐channel if soft seam
  CudaStatus status_;
};

} // namespace cuda

std::optional<cv::Scalar> match_seam_images3(
    cv::Mat& image0,
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam_color, // 3‐channel seam mask
    int N,
    const cv::Point& topLeft0,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose = false);

} // namespace pano
} // namespace hm

#include "cudaPano3.inl"
