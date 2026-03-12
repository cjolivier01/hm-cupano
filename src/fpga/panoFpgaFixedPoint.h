#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace hm {
namespace fpga {
namespace fixed {

constexpr uint16_t kBlendPixelBits = 18;
constexpr uint16_t kBlendFractionBits = 8;
constexpr uint16_t kBlendIntegerBits = kBlendPixelBits - kBlendFractionBits - 1;
constexpr uint16_t kMaskBits = 16;
constexpr uint16_t kMaskFractionBits = 16;
constexpr uint16_t kAdjustBits = 16;
constexpr uint16_t kAdjustFractionBits = 8;

constexpr int32_t kBlendMinQ9_8 = -(1 << (kBlendPixelBits - 1));
constexpr int32_t kBlendMaxQ9_8 = (1 << (kBlendPixelBits - 1)) - 1;
constexpr uint32_t kMaskOneU0_16 = (1u << kMaskBits) - 1u;

inline int32_t EncodePixelQ9_8(float value) {
  const float scaled = std::round(value * static_cast<float>(1 << kBlendFractionBits));
  return static_cast<int32_t>(std::clamp(scaled, static_cast<float>(kBlendMinQ9_8), static_cast<float>(kBlendMaxQ9_8)));
}

inline uint16_t EncodeMaskU0_16(uint8_t mask_value) {
  return mask_value == 0 ? 0u : static_cast<uint16_t>(kMaskOneU0_16);
}

inline int16_t EncodeAdjustQ8_8(float value) {
  const float scaled = std::round(value * static_cast<float>(1 << kAdjustFractionBits));
  return static_cast<int16_t>(std::clamp(scaled, -32768.0f, 32767.0f));
}

} // namespace fixed
} // namespace fpga
} // namespace hm
