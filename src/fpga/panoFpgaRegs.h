#pragma once

#include "cupano/fpga/panoFpgaTypes.h"

#include <cstdint>

namespace hm {
namespace fpga {
namespace reg {

constexpr uint32_t kControl = 0x00;
constexpr uint32_t kStatus = 0x04;
constexpr uint32_t kIrqEnable = 0x08;
constexpr uint32_t kOpcode = 0x0c;
constexpr uint32_t kFlags = 0x10;

constexpr uint32_t kSrc0AddrLo = 0x20;
constexpr uint32_t kSrc0AddrHi = 0x24;
constexpr uint32_t kSrc0Stride = 0x28;
constexpr uint32_t kSrc0Extent = 0x2c;
constexpr uint32_t kSrc0Format = 0x30;

constexpr uint32_t kSrc1AddrLo = 0x40;
constexpr uint32_t kSrc1AddrHi = 0x44;
constexpr uint32_t kSrc1Stride = 0x48;
constexpr uint32_t kSrc1Extent = 0x4c;
constexpr uint32_t kSrc1Format = 0x50;

constexpr uint32_t kSrc2AddrLo = 0x60;
constexpr uint32_t kSrc2AddrHi = 0x64;
constexpr uint32_t kSrc2Stride = 0x68;
constexpr uint32_t kSrc2Extent = 0x6c;
constexpr uint32_t kSrc2Format = 0x70;

constexpr uint32_t kDestAddrLo = 0x80;
constexpr uint32_t kDestAddrHi = 0x84;
constexpr uint32_t kDestStride = 0x88;
constexpr uint32_t kDestExtent = 0x8c;
constexpr uint32_t kDestFormat = 0x90;

constexpr uint32_t kMapXAddrLo = 0xa0;
constexpr uint32_t kMapXAddrHi = 0xa4;
constexpr uint32_t kMapYAddrLo = 0xa8;
constexpr uint32_t kMapYAddrHi = 0xac;
constexpr uint32_t kRemapExtent = 0xb0;
constexpr uint32_t kRemapOffset = 0xb4;
constexpr uint32_t kRemapFlags = 0xb8;
constexpr uint32_t kAdjust01 = 0xbc;
constexpr uint32_t kAdjust2 = 0xc0;

constexpr uint32_t kCopySrcXY = 0xc4;
constexpr uint32_t kCopyDestXY = 0xc8;
constexpr uint32_t kCopyExtent = 0xcc;

constexpr uint32_t kBlendExtent = 0xd0;
constexpr uint32_t kBlendLevels = 0xd4;
constexpr uint32_t kBlendCfg = 0xd8;

constexpr uint32_t kPyramidLowExtent = 0xdc;
constexpr uint32_t kPyramidHighExtent = 0xe0;
constexpr uint32_t kPyramidCfg = 0xe4;

namespace bit {
constexpr uint32_t kStart = 1u << 0;
constexpr uint32_t kSoftReset = 1u << 1;
constexpr uint32_t kIrqAck = 1u << 2;

constexpr uint32_t kBusy = 1u << 0;
constexpr uint32_t kDone = 1u << 1;
constexpr uint32_t kError = 1u << 2;
constexpr uint32_t kTimeout = 1u << 3;
} // namespace bit

inline constexpr uint32_t pack_extent(uint32_t width, uint32_t height) {
  return ((height & 0xffffu) << 16) | (width & 0xffffu);
}

inline constexpr uint32_t pack_xy(uint32_t x, uint32_t y) {
  return ((y & 0xffffu) << 16) | (x & 0xffffu);
}

inline constexpr uint32_t pack_signed_xy(int32_t x, int32_t y) {
  return ((static_cast<uint32_t>(y) & 0xffffu) << 16) | (static_cast<uint32_t>(x) & 0xffffu);
}

inline constexpr uint32_t pack_adjust01(int16_t x, int16_t y) {
  return ((static_cast<uint32_t>(static_cast<uint16_t>(y))) << 16) | static_cast<uint32_t>(static_cast<uint16_t>(x));
}

inline constexpr uint32_t pack_format(PixelFormat format) {
  return static_cast<uint32_t>(format);
}

} // namespace reg
} // namespace fpga
} // namespace hm
