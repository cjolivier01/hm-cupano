#pragma once

#include <cstdint>

namespace hm {
namespace fpga {

enum class PanoAccelOpcode : uint32_t {
  kNoop = 0,
  kRemap = 1,
  kCopyRoi = 2,
  kPyramidBlend = 3,
  kDownsampleLevel = 4,
  kLaplacianLevel = 5,
  kBlendLevel = 6,
  kReconstructLevel = 7,
};

enum class PixelFormat : uint32_t {
  kGray8 = 0x11,
  kGray16 = 0x12,
  kRgb888 = 0x23,
  kRgba8888 = 0x24,
  kGrayF32 = 0x31,
  kRgbF32 = 0x33,
  kRgbaF32 = 0x34,
};

inline constexpr uint32_t channels(PixelFormat format) {
  switch (format) {
    case PixelFormat::kGray8:
    case PixelFormat::kGray16:
    case PixelFormat::kGrayF32:
      return 1;
    case PixelFormat::kRgb888:
    case PixelFormat::kRgbF32:
      return 3;
    case PixelFormat::kRgba8888:
    case PixelFormat::kRgbaF32:
      return 4;
  }
  return 0;
}

inline constexpr uint32_t bytes_per_channel(PixelFormat format) {
  switch (format) {
    case PixelFormat::kGray8:
    case PixelFormat::kRgb888:
    case PixelFormat::kRgba8888:
      return 1;
    case PixelFormat::kGray16:
      return 2;
    case PixelFormat::kGrayF32:
    case PixelFormat::kRgbF32:
    case PixelFormat::kRgbaF32:
      return 4;
  }
  return 0;
}

inline constexpr uint32_t bytes_per_pixel(PixelFormat format) {
  return channels(format) * bytes_per_channel(format);
}

inline constexpr bool format_has_alpha(PixelFormat format) {
  switch (format) {
    case PixelFormat::kRgba8888:
    case PixelFormat::kRgbaF32:
      return true;
    default:
      return false;
  }
}

enum class LogicalBuffer : uint32_t {
  kInput1 = 0,
  kInput2 = 1,
  kCanvas = 2,
  kBlendMask = 3,
  kBlendLeft = 4,
  kBlendRight = 5,
  kBlendOutput = 6,
  kRemap1X = 7,
  kRemap1Y = 8,
  kRemap2X = 9,
  kRemap2Y = 10,
};

struct BufferGeometry {
  LogicalBuffer id{LogicalBuffer::kInput1};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t stride_bytes{0};
  PixelFormat format{PixelFormat::kRgb888};
};

struct BufferBinding {
  LogicalBuffer id{LogicalBuffer::kInput1};
  uint64_t phys_addr{0};
};

struct ImageBuffer {
  uint64_t phys_addr{0};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t stride_bytes{0};
  PixelFormat format{PixelFormat::kRgb888};
};

struct RemapConfig {
  uint64_t map_x_addr{0};
  uint64_t map_y_addr{0};
  uint32_t map_width{0};
  uint32_t map_height{0};
  int32_t offset_x{0};
  int32_t offset_y{0};
  bool no_unmapped_write{false};
  int16_t adjust_r_q8_8{0};
  int16_t adjust_g_q8_8{0};
  int16_t adjust_b_q8_8{0};
};

struct CopyRoiConfig {
  uint32_t src_x{0};
  uint32_t src_y{0};
  uint32_t width{0};
  uint32_t height{0};
  uint32_t dest_x{0};
  uint32_t dest_y{0};
};

struct BlendConfig {
  uint32_t width{0};
  uint32_t height{0};
  uint32_t num_levels{0};
  uint16_t channels{0};
  uint16_t fixed_point_fraction_bits{8};
  bool alpha_aware{false};
};

struct PyramidLevelConfig {
  uint32_t low_width{0};
  uint32_t low_height{0};
  uint32_t high_width{0};
  uint32_t high_height{0};
  uint16_t channels{0};
  uint16_t fixed_point_fraction_bits{8};
  bool alpha_aware{false};
};

struct PanoOperation {
  PanoAccelOpcode opcode{PanoAccelOpcode::kNoop};
  uint32_t flags{0};
  ImageBuffer src0;
  ImageBuffer src1;
  ImageBuffer src2;
  ImageBuffer dest;
  RemapConfig remap;
  CopyRoiConfig copy;
  BlendConfig blend;
  PyramidLevelConfig pyramid;
  uint32_t timeout_ms{1000};
};

} // namespace fpga
} // namespace hm
