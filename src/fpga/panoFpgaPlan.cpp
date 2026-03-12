#include "cupano/fpga/panoFpgaPlan.h"

#include "cupano/fpga/panoFpgaFixedPoint.h"
#include "cupano/pano/canvasManager.h"
#include "cupano/pano/controlMasks.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <stdexcept>

namespace hm {
namespace fpga {
namespace {

BufferGeometry make_buffer(LogicalBuffer id, uint32_t width, uint32_t height, PixelFormat format) {
  return BufferGeometry{
      .id = id,
      .width = width,
      .height = height,
      .stride_bytes = width * bytes_per_pixel(format),
      .format = format,
  };
}

const BufferGeometry& geometry_for(const StitchPlan& plan, LogicalBuffer id) {
  const auto it = std::find_if(
      plan.buffers.begin(), plan.buffers.end(), [&](const BufferGeometry& geometry) { return geometry.id == id; });
  if (it == plan.buffers.end()) {
    throw std::invalid_argument(std::string("No geometry for logical buffer ") + LogicalBufferName(id));
  }
  return *it;
}

uint64_t binding_for(const std::vector<BufferBinding>& bindings, LogicalBuffer id) {
  const auto it =
      std::find_if(bindings.begin(), bindings.end(), [&](const BufferBinding& binding) { return binding.id == id; });
  if (it == bindings.end()) {
    throw std::invalid_argument(std::string("No physical binding for logical buffer ") + LogicalBufferName(id));
  }
  return it->phys_addr;
}

ImageBuffer bind_image(const StitchPlan& plan, const std::vector<BufferBinding>& bindings, LogicalBuffer id) {
  const BufferGeometry& geometry = geometry_for(plan, id);
  return ImageBuffer{
      .phys_addr = binding_for(bindings, id),
      .width = geometry.width,
      .height = geometry.height,
      .stride_bytes = geometry.stride_bytes,
      .format = geometry.format,
  };
}

} // namespace

const char* LogicalBufferName(LogicalBuffer id) {
  switch (id) {
    case LogicalBuffer::kInput1:
      return "input1";
    case LogicalBuffer::kInput2:
      return "input2";
    case LogicalBuffer::kCanvas:
      return "canvas";
    case LogicalBuffer::kBlendMask:
      return "blend_mask";
    case LogicalBuffer::kBlendLeft:
      return "blend_left";
    case LogicalBuffer::kBlendRight:
      return "blend_right";
    case LogicalBuffer::kBlendOutput:
      return "blend_output";
    case LogicalBuffer::kRemap1X:
      return "remap1_x";
    case LogicalBuffer::kRemap1Y:
      return "remap1_y";
    case LogicalBuffer::kRemap2X:
      return "remap2_x";
    case LogicalBuffer::kRemap2Y:
      return "remap2_y";
  }
  return "unknown";
}

StitchPlan BuildTwoCameraSoftSeamPlan(
    const hm::pano::ControlMasks& control_masks,
    int num_levels,
    PixelFormat pipeline_format,
    uint32_t overlap_padding) {
  const bool masks_are_valid = !control_masks.img1_col.empty() && !control_masks.img1_row.empty() &&
      !control_masks.img2_col.empty() && !control_masks.img2_row.empty() &&
      !control_masks.whole_seam_mask_image.empty() && control_masks.positions.size() == 2;
  if (!masks_are_valid) {
    throw std::invalid_argument("ControlMasks must be populated before building an FPGA stitch plan");
  }
  if (num_levels <= 0) {
    throw std::invalid_argument("Soft-seam FPGA plan requires num_levels > 0");
  }

  const int canvas_width = static_cast<int>(std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols));
  const int canvas_height = static_cast<int>(std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows));

  hm::pano::CanvasManager canvas_manager(
      hm::pano::CanvasInfo{
          .width = canvas_width,
          .height = canvas_height,
          .positions =
              {
                  cv::Point(
                      static_cast<int>(control_masks.positions[0].xpos),
                      static_cast<int>(control_masks.positions[0].ypos)),
                  cv::Point(
                      static_cast<int>(control_masks.positions[1].xpos),
                      static_cast<int>(control_masks.positions[1].ypos)),
              },
      },
      /*minimize_blend=*/true,
      static_cast<int>(overlap_padding));

  canvas_manager._remapper_1.width = control_masks.img1_col.cols;
  canvas_manager._remapper_1.height = control_masks.img1_col.rows;
  canvas_manager._remapper_2.width = control_masks.img2_col.cols;
  canvas_manager._remapper_2.height = control_masks.img2_col.rows;
  canvas_manager.updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  const cv::Mat blend_mask = canvas_manager.convertMaskMat(control_masks.whole_seam_mask_image);
  if (blend_mask.empty()) {
    throw std::runtime_error("CanvasManager produced an empty blend mask");
  }

  StitchPlan plan;
  plan.canvas_width = static_cast<uint32_t>(canvas_manager.canvas_width());
  plan.canvas_height = static_cast<uint32_t>(canvas_manager.canvas_height());
  plan.buffers = {
      make_buffer(LogicalBuffer::kInput1, control_masks.img1_col.cols, control_masks.img1_col.rows, pipeline_format),
      make_buffer(LogicalBuffer::kInput2, control_masks.img2_col.cols, control_masks.img2_col.rows, pipeline_format),
      make_buffer(LogicalBuffer::kCanvas, plan.canvas_width, plan.canvas_height, pipeline_format),
      make_buffer(LogicalBuffer::kBlendMask, blend_mask.cols, blend_mask.rows, PixelFormat::kGray16),
      make_buffer(LogicalBuffer::kBlendLeft, blend_mask.cols, blend_mask.rows, pipeline_format),
      make_buffer(LogicalBuffer::kBlendRight, blend_mask.cols, blend_mask.rows, pipeline_format),
      make_buffer(LogicalBuffer::kBlendOutput, blend_mask.cols, blend_mask.rows, pipeline_format),
      make_buffer(
          LogicalBuffer::kRemap1X, control_masks.img1_col.cols, control_masks.img1_col.rows, PixelFormat::kGray16),
      make_buffer(
          LogicalBuffer::kRemap1Y, control_masks.img1_row.cols, control_masks.img1_row.rows, PixelFormat::kGray16),
      make_buffer(
          LogicalBuffer::kRemap2X, control_masks.img2_col.cols, control_masks.img2_col.rows, PixelFormat::kGray16),
      make_buffer(
          LogicalBuffer::kRemap2Y, control_masks.img2_row.cols, control_masks.img2_row.rows, PixelFormat::kGray16),
  };

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kRemap,
      .src0 = LogicalBuffer::kInput1,
      .dest = LogicalBuffer::kCanvas,
      .map_x = LogicalBuffer::kRemap1X,
      .map_y = LogicalBuffer::kRemap1Y,
      .remap =
          RemapConfig{
              .map_width = static_cast<uint32_t>(control_masks.img1_col.cols),
              .map_height = static_cast<uint32_t>(control_masks.img1_col.rows),
              .offset_x = canvas_manager._x1,
              .offset_y = canvas_manager._y1,
              .no_unmapped_write = false,
          },
  });

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kCopyRoi,
      .src0 = LogicalBuffer::kCanvas,
      .dest = LogicalBuffer::kBlendLeft,
      .copy =
          CopyRoiConfig{
              .src_x = static_cast<uint32_t>(canvas_manager.remapped_image_roi_blend_1.x),
              .src_y = 0,
              .width = static_cast<uint32_t>(canvas_manager.remapped_image_roi_blend_1.width),
              .height = static_cast<uint32_t>(blend_mask.rows),
              .dest_x = static_cast<uint32_t>(canvas_manager._remapper_1.xpos),
              .dest_y = 0,
          },
  });

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kRemap,
      .src0 = LogicalBuffer::kInput2,
      .dest = LogicalBuffer::kCanvas,
      .map_x = LogicalBuffer::kRemap2X,
      .map_y = LogicalBuffer::kRemap2Y,
      .remap =
          RemapConfig{
              .map_width = static_cast<uint32_t>(control_masks.img2_col.cols),
              .map_height = static_cast<uint32_t>(control_masks.img2_col.rows),
              .offset_x = canvas_manager._x2,
              .offset_y = canvas_manager._y2,
              .no_unmapped_write = false,
          },
  });

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kCopyRoi,
      .src0 = LogicalBuffer::kCanvas,
      .dest = LogicalBuffer::kBlendRight,
      .copy =
          CopyRoiConfig{
              .src_x = static_cast<uint32_t>(canvas_manager._x2),
              .src_y = 0,
              .width = static_cast<uint32_t>(canvas_manager.remapped_image_roi_blend_2.width),
              .height = static_cast<uint32_t>(blend_mask.rows),
              .dest_x = static_cast<uint32_t>(canvas_manager._remapper_2.xpos),
              .dest_y = 0,
          },
  });

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kPyramidBlend,
      .src0 = LogicalBuffer::kBlendLeft,
      .src1 = LogicalBuffer::kBlendRight,
      .src2 = LogicalBuffer::kBlendMask,
      .dest = LogicalBuffer::kBlendOutput,
      .blend =
          BlendConfig{
              .width = static_cast<uint32_t>(blend_mask.cols),
              .height = static_cast<uint32_t>(blend_mask.rows),
              .num_levels = static_cast<uint32_t>(num_levels),
              .channels = static_cast<uint16_t>(channels(pipeline_format)),
              .fixed_point_fraction_bits = fixed::kBlendFractionBits,
              .alpha_aware = format_has_alpha(pipeline_format),
          },
  });

  plan.stages.push_back(PlannedStage{
      .opcode = PanoAccelOpcode::kCopyRoi,
      .src0 = LogicalBuffer::kBlendOutput,
      .dest = LogicalBuffer::kCanvas,
      .copy =
          CopyRoiConfig{
              .src_x = 0,
              .src_y = 0,
              .width = static_cast<uint32_t>(blend_mask.cols),
              .height = static_cast<uint32_t>(blend_mask.rows),
              .dest_x = static_cast<uint32_t>(canvas_manager._x2 - canvas_manager.overlap_padding()),
              .dest_y = 0,
          },
  });

  return plan;
}

std::vector<PanoOperation> BindPlan(const StitchPlan& plan, const std::vector<BufferBinding>& bindings) {
  std::vector<PanoOperation> bound;
  bound.reserve(plan.stages.size());
  for (const PlannedStage& stage : plan.stages) {
    PanoOperation operation;
    operation.opcode = stage.opcode;
    operation.flags = stage.flags;
    operation.timeout_ms = stage.timeout_ms;

    switch (stage.opcode) {
      case PanoAccelOpcode::kRemap:
        operation.src0 = bind_image(plan, bindings, stage.src0);
        operation.dest = bind_image(plan, bindings, stage.dest);
        operation.remap = stage.remap;
        operation.remap.map_x_addr = binding_for(bindings, stage.map_x);
        operation.remap.map_y_addr = binding_for(bindings, stage.map_y);
        break;
      case PanoAccelOpcode::kCopyRoi:
        operation.src0 = bind_image(plan, bindings, stage.src0);
        operation.dest = bind_image(plan, bindings, stage.dest);
        operation.copy = stage.copy;
        break;
      case PanoAccelOpcode::kPyramidBlend:
        operation.src0 = bind_image(plan, bindings, stage.src0);
        operation.src1 = bind_image(plan, bindings, stage.src1);
        operation.src2 = bind_image(plan, bindings, stage.src2);
        operation.dest = bind_image(plan, bindings, stage.dest);
        operation.blend = stage.blend;
        break;
      default:
        throw std::invalid_argument("BindPlan only supports remap, copy, and pyramid blend stages");
    }

    bound.push_back(operation);
  }
  return bound;
}

} // namespace fpga
} // namespace hm
