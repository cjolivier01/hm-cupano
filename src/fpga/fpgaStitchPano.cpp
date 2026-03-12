#include "cupano/fpga/fpgaStitchPano.h"

#include "cupano/fpga/panoFpgaFixedPoint.h"
#include "cupano/pano/controlMasks.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace hm {
namespace fpga {
namespace {

const BufferGeometry& FindGeometry(const StitchPlan& plan, LogicalBuffer id) {
  const auto it = std::find_if(
      plan.buffers.begin(), plan.buffers.end(), [&](const BufferGeometry& geometry) { return geometry.id == id; });
  if (it == plan.buffers.end()) {
    throw std::invalid_argument(std::string("Missing plan geometry for ") + LogicalBufferName(id));
  }
  return *it;
}

const PlannedStage& FindBlendPasteStage(const StitchPlan& plan) {
  const auto it = std::find_if(plan.stages.begin(), plan.stages.end(), [](const PlannedStage& stage) {
    return stage.opcode == PanoAccelOpcode::kCopyRoi && stage.src0 == LogicalBuffer::kBlendOutput &&
        stage.dest == LogicalBuffer::kCanvas;
  });
  if (it == plan.stages.end()) {
    throw std::invalid_argument("Missing blend-output paste stage in FPGA stitch plan");
  }
  return *it;
}

FpgaStatus ValidateContinuousU8(const cv::Mat& matrix, const char* name) {
  if (matrix.empty()) {
    return FpgaStatus(std::string(name) + " must not be empty");
  }
  if (matrix.type() != CV_8U) {
    return FpgaStatus(std::string(name) + " must be CV_8U");
  }
  if (!matrix.isContinuous()) {
    return FpgaStatus(std::string(name) + " must be continuous");
  }
  return FpgaStatus::OkStatus();
}

} // namespace

FpgaStitchPano::FpgaStitchPano(
    int batch_size,
    int num_levels,
    const hm::pano::ControlMasks& control_masks,
    PhysicalBufferAllocator& allocator,
    PanoOperationExecutor& executor,
    PixelFormat pipeline_format,
    bool quiet,
    FpgaBlendProfile blend_profile,
    uint32_t overlap_padding)
    : batch_size_(batch_size),
      pipeline_format_(pipeline_format),
      quiet_(quiet),
      blend_profile_(blend_profile),
      overlap_padding_(overlap_padding),
      allocator_(allocator),
      executor_(executor) {
  if (batch_size_ != 1) {
    status_ = FpgaStatus("FpgaStitchPano currently supports batch_size == 1 only");
    return;
  }
  if (pipeline_format_ != PixelFormat::kRgba8888 && pipeline_format_ != PixelFormat::kRgb888) {
    status_ = FpgaStatus("FpgaStitchPano currently supports RGB888 or RGBA8888 frame boundaries");
    return;
  }

  try {
    plan_ = BuildTwoCameraSoftSeamPlan(control_masks, num_levels, pipeline_format_, overlap_padding);
  } catch (const std::exception& ex) {
    status_ = FpgaStatus(ex.what());
    return;
  }

  status_ = initialize_buffers(control_masks);
  if (status_.ok() && !quiet_) {
    std::cout << "FPGA stitched canvas size: " << plan_.canvas_width << " x " << plan_.canvas_height << std::endl;
  }
}

FpgaStatus FpgaStitchPano::initialize_buffers(const hm::pano::ControlMasks& control_masks) {
  for (const BufferGeometry& geometry : plan_.buffers) {
    if (geometry.id == LogicalBuffer::kInput1 || geometry.id == LogicalBuffer::kInput2) {
      continue;
    }
    PhysicalBuffer allocation;
    FPGA_ASSIGN_OR_RETURN(allocation, allocator_.Allocate(geometry.id, geometry));
    buffers_[geometry.id] = allocation;
  }

  FPGA_RETURN_IF_ERROR(upload_matrix_as_bytes(LogicalBuffer::kRemap1X, control_masks.img1_col));
  FPGA_RETURN_IF_ERROR(upload_matrix_as_bytes(LogicalBuffer::kRemap1Y, control_masks.img1_row));
  FPGA_RETURN_IF_ERROR(upload_matrix_as_bytes(LogicalBuffer::kRemap2X, control_masks.img2_col));
  FPGA_RETURN_IF_ERROR(upload_matrix_as_bytes(LogicalBuffer::kRemap2Y, control_masks.img2_row));
  FPGA_RETURN_IF_ERROR(upload_mask_u0_16(control_masks.whole_seam_mask_image));

  return FpgaStatus::OkStatus();
}

FpgaStatus FpgaStitchPano::upload_matrix_as_bytes(LogicalBuffer id, const cv::Mat& matrix) {
  const auto it = buffers_.find(id);
  if (it == buffers_.end()) {
    return FpgaStatus(std::string("Missing physical buffer for ") + LogicalBufferName(id));
  }
  if (matrix.empty()) {
    return FpgaStatus(std::string("Matrix upload source is empty for ") + LogicalBufferName(id));
  }
  if (!matrix.isContinuous()) {
    return FpgaStatus(std::string("Matrix upload source must be continuous for ") + LogicalBufferName(id));
  }
  return allocator_.CopyFromHost(it->second, matrix.data, matrix.total() * matrix.elemSize(), 0);
}

FpgaStatus FpgaStitchPano::upload_mask_u0_16(const cv::Mat& full_mask_u8) {
  FPGA_RETURN_IF_ERROR(ValidateContinuousU8(full_mask_u8, "Blend mask"));

  const BufferGeometry& mask_geometry = FindGeometry(plan_, LogicalBuffer::kBlendMask);
  const PlannedStage& paste_stage = FindBlendPasteStage(plan_);
  const uint32_t crop_x = paste_stage.copy.dest_x;
  const uint32_t crop_y = paste_stage.copy.dest_y;

  if (crop_x + mask_geometry.width > static_cast<uint32_t>(plan_.canvas_width) ||
      crop_y + mask_geometry.height > static_cast<uint32_t>(plan_.canvas_height)) {
    return FpgaStatus("Blend mask crop exceeds canvas bounds");
  }

  std::vector<uint16_t> encoded(static_cast<size_t>(mask_geometry.width) * static_cast<size_t>(mask_geometry.height));
  for (uint32_t y = 0; y < mask_geometry.height; ++y) {
    const uint32_t src_y = std::min<uint32_t>(crop_y + y, static_cast<uint32_t>(full_mask_u8.rows - 1));
    for (uint32_t x = 0; x < mask_geometry.width; ++x) {
      const uint32_t src_x = std::min<uint32_t>(crop_x + x, static_cast<uint32_t>(full_mask_u8.cols - 1));
      const uint8_t value =
          full_mask_u8.data[static_cast<size_t>(src_y) * static_cast<size_t>(full_mask_u8.cols) + src_x];
      encoded[static_cast<size_t>(y) * static_cast<size_t>(mask_geometry.width) + x] = fixed::EncodeMaskU0_16(value);
    }
  }
  const auto it = buffers_.find(LogicalBuffer::kBlendMask);
  if (it == buffers_.end()) {
    return FpgaStatus("Missing physical buffer for blend mask");
  }
  return allocator_.CopyFromHost(it->second, encoded.data(), encoded.size() * sizeof(uint16_t), 0);
}

FpgaStatus FpgaStitchPano::validate_input(const ImageBuffer& input, LogicalBuffer logical_buffer) const {
  auto geometry_it = std::find_if(plan_.buffers.begin(), plan_.buffers.end(), [&](const BufferGeometry& geometry) {
    return geometry.id == logical_buffer;
  });
  if (geometry_it == plan_.buffers.end()) {
    return FpgaStatus(std::string("Missing plan geometry for ") + LogicalBufferName(logical_buffer));
  }
  if (input.width != geometry_it->width || input.height != geometry_it->height) {
    return FpgaStatus(std::string("Input dimensions do not match plan for ") + LogicalBufferName(logical_buffer));
  }
  if (input.format != geometry_it->format) {
    return FpgaStatus(std::string("Input format does not match plan for ") + LogicalBufferName(logical_buffer));
  }
  return FpgaStatus::OkStatus();
}

FpgaStatus FpgaStitchPano::clear_run_buffers() {
  for (LogicalBuffer id :
       {LogicalBuffer::kCanvas, LogicalBuffer::kBlendLeft, LogicalBuffer::kBlendRight, LogicalBuffer::kBlendOutput}) {
    const auto it = buffers_.find(id);
    if (it != buffers_.end()) {
      FPGA_RETURN_IF_ERROR(allocator_.Memset(it->second, 0));
    }
  }
  return FpgaStatus::OkStatus();
}

std::vector<BufferBinding> FpgaStitchPano::bindings_for_process(
    const ImageBuffer& input_image1,
    const ImageBuffer& input_image2) const {
  std::vector<BufferBinding> bindings;
  bindings.push_back(BufferBinding{.id = LogicalBuffer::kInput1, .phys_addr = input_image1.phys_addr});
  bindings.push_back(BufferBinding{.id = LogicalBuffer::kInput2, .phys_addr = input_image2.phys_addr});
  for (const auto& item : buffers_) {
    bindings.push_back(BufferBinding{.id = item.first, .phys_addr = item.second.image.phys_addr});
  }
  return bindings;
}

FpgaStatusOr<ImageBuffer> FpgaStitchPano::process(const ImageBuffer& input_image1, const ImageBuffer& input_image2) {
  if (!status_.ok()) {
    return status_;
  }
  FPGA_RETURN_IF_ERROR(validate_input(input_image1, LogicalBuffer::kInput1));
  FPGA_RETURN_IF_ERROR(validate_input(input_image2, LogicalBuffer::kInput2));
  FPGA_RETURN_IF_ERROR(clear_run_buffers());

  const std::vector<PanoOperation> operations = BindPlan(plan_, bindings_for_process(input_image1, input_image2));
  for (const PanoOperation& operation : operations) {
    const RunResult result = executor_.Run(operation, std::chrono::milliseconds(operation.timeout_ms));
    if (!result.ok) {
      return FpgaStatus("FPGA operation failed: " + result.message);
    }
  }

  return buffers_.at(LogicalBuffer::kCanvas).image;
}

std::optional<PhysicalBuffer> FpgaStitchPano::lookup_buffer(LogicalBuffer id) const {
  const auto it = buffers_.find(id);
  if (it == buffers_.end()) {
    return std::nullopt;
  }
  return it->second;
}

} // namespace fpga
} // namespace hm
