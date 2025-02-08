#pragma once

#include "canvasManager.h"
#include "controlMasks.h"
#include "cudaBlend.h"
#include "cudaMat.h"
#include "cudaStatus.h"

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
 */
template <typename T_pipeline, typename T_compute>
struct StitchingContext {
  StitchingContext(int batch_size, bool is_hard_seam) : batch_size_(batch_size), is_hard_seam_(is_hard_seam) {}
  // Static buffers
  std::unique_ptr<CudaMat<uint16_t>> remap_1_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_1_y;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_x;
  std::unique_ptr<CudaMat<uint16_t>> remap_2_y;

  std::unique_ptr<CudaMat<T_compute>> cudaBlendSoftSeam;
  std::unique_ptr<CudaMat<unsigned char>> cudaBlendHardSeam;

  // Scratch buffers
  std::unique_ptr<CudaMat<T_compute>> cudaFull1;
  std::unique_ptr<CudaMat<T_compute>> cudaFull2;

  // Laplacian Blend Scratch context
  std::unique_ptr<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>> laplacian_blend_context;

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
 *
 */
template <typename T_pipeline, typename T_compute>
class CudaStitchPano {
 public:
  CudaStitchPano(int batch_size, int num_levels, const ControlMasks& control_masks, bool match_exposure = true);

  int canvas_width() const {
    return canvas_manager_->canvas_width();
  }

  int canvas_height() const {
    return canvas_manager_->canvas_height();
  }

  int batch_size() const {
    return stitch_context_->batch_size();
  }

  CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process(
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 protected:
  static CudaStatusOr<std::unique_ptr<CudaMat<T_pipeline>>> process(
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2,
      StitchingContext<T_pipeline, T_compute>& stitch_context,
      const CanvasManager& canvas_manager,
      const std::optional<T_compute>& image_adjustment,
      cudaStream_t stream,
      std::unique_ptr<CudaMat<T_pipeline>>&& canvas);

 private:
  std::optional<float3> compute_image_adjustment(
      const CudaMat<T_pipeline>& inputImage1,
      const CudaMat<T_pipeline>& inputImage2);

  std::unique_ptr<StitchingContext<T_pipeline, T_compute>> stitch_context_;
  std::unique_ptr<CanvasManager> canvas_manager_;
  bool match_exposure_;
  std::optional<float3> image_adjustment_;
  std::optional<cv::Mat> whole_seam_mask_image_;
};

} // namespace cuda

std::optional<cv::Scalar> match_seam_images(
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam,
    int N,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose = false);

} // namespace pano
} // namespace hm

#include "cudaPano.inl"
