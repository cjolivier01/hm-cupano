#pragma once

#include "cupano/fpga/panoFpgaPlan.h"
#include "cupano/fpga/panoFpgaRuntime.h"
#include "cupano/fpga/panoFpgaStatus.h"

#include <opencv2/core.hpp>

#include <map>
#include <optional>

namespace hm {
namespace pano {
class ControlMasks;
}
namespace fpga {

enum class FpgaBlendProfile : uint32_t {
  kQ9_8 = 0,
};

class FpgaStitchPano {
 public:
  FpgaStitchPano(
      int batch_size,
      int num_levels,
      const hm::pano::ControlMasks& control_masks,
      PhysicalBufferAllocator& allocator,
      PanoOperationExecutor& executor,
      PixelFormat pipeline_format = PixelFormat::kRgba8888,
      bool quiet = false,
      FpgaBlendProfile blend_profile = FpgaBlendProfile::kQ9_8,
      uint32_t overlap_padding = 128);

  int canvas_width() const {
    return static_cast<int>(plan_.canvas_width);
  }

  int canvas_height() const {
    return static_cast<int>(plan_.canvas_height);
  }

  int batch_size() const {
    return batch_size_;
  }

  const FpgaStatus& status() const {
    return status_;
  }

  const StitchPlan& plan() const {
    return plan_;
  }

  FpgaStatusOr<ImageBuffer> process(const ImageBuffer& input_image1, const ImageBuffer& input_image2);
  std::optional<PhysicalBuffer> lookup_buffer(LogicalBuffer id) const;

 private:
  FpgaStatus initialize_buffers(const hm::pano::ControlMasks& control_masks);
  FpgaStatus validate_input(const ImageBuffer& input, LogicalBuffer logical_buffer) const;
  FpgaStatus clear_run_buffers();
  FpgaStatus upload_matrix_as_bytes(LogicalBuffer id, const cv::Mat& matrix);
  FpgaStatus upload_mask_u0_16(const cv::Mat& mask_u8);
  std::vector<BufferBinding> bindings_for_process(const ImageBuffer& input_image1, const ImageBuffer& input_image2)
      const;

  int batch_size_{0};
  PixelFormat pipeline_format_{PixelFormat::kRgba8888};
  bool quiet_{false};
  FpgaBlendProfile blend_profile_{FpgaBlendProfile::kQ9_8};
  uint32_t overlap_padding_{128};
  PhysicalBufferAllocator& allocator_;
  PanoOperationExecutor& executor_;
  StitchPlan plan_;
  std::map<LogicalBuffer, PhysicalBuffer> buffers_;
  FpgaStatus status_;
};

} // namespace fpga
} // namespace hm
