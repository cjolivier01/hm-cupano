#pragma once

#include "cupano/fpga/panoFpgaTypes.h"

#include <cstdint>
#include <vector>

namespace hm {
namespace pano {
class ControlMasks;
}
namespace fpga {

const char* LogicalBufferName(LogicalBuffer id);

struct PlannedStage {
  PanoAccelOpcode opcode{PanoAccelOpcode::kNoop};
  uint32_t flags{0};
  LogicalBuffer src0{LogicalBuffer::kInput1};
  LogicalBuffer src1{LogicalBuffer::kInput2};
  LogicalBuffer src2{LogicalBuffer::kBlendMask};
  LogicalBuffer dest{LogicalBuffer::kCanvas};
  LogicalBuffer map_x{LogicalBuffer::kRemap1X};
  LogicalBuffer map_y{LogicalBuffer::kRemap1Y};
  RemapConfig remap;
  CopyRoiConfig copy;
  BlendConfig blend;
  uint32_t timeout_ms{1000};
};

struct StitchPlan {
  uint32_t canvas_width{0};
  uint32_t canvas_height{0};
  std::vector<BufferGeometry> buffers;
  std::vector<PlannedStage> stages;
};

StitchPlan BuildTwoCameraSoftSeamPlan(
    const hm::pano::ControlMasks& control_masks,
    int num_levels,
    PixelFormat pipeline_format,
    uint32_t overlap_padding = 128);

std::vector<PanoOperation> BindPlan(const StitchPlan& plan, const std::vector<BufferBinding>& bindings);

} // namespace fpga
} // namespace hm
