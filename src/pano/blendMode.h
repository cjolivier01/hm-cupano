#pragma once

#include <cstdint>
#include <string>

namespace hm {
namespace pano {

/**
 * @brief Which blend operator a stitcher uses to combine remapped cameras.
 */
enum class BlendMode : uint8_t {
  kHardSeam = 0, ///< Per-pixel owner label, no mixing.
  kLaplacian = 1, ///< Multi-band pyramid blend.
  kAlpha = 2, ///< Pointwise feathered crossfade at full resolution.
};

/**
 * @brief Blend operator configuration shared by CudaStitchPano, CudaStitchPano3 and CudaStitchPanoN.
 *
 * This describes the blend operator only. Batch size, quiet, minimize_blend, max_output_width and
 * compact_workspace remain separate stitcher options.
 *
 * The three stitcher constructors historically took an `int num_levels` in which `0` meant "hard
 * seam". The implicit constructor below preserves that encoding for every existing caller, and is
 * the single place where it is interpreted.
 */
struct BlendSettings {
  /// Largest accepted feather fraction. Callers driving a UI should offer a tighter range: a wide
  /// feather reintroduces the wide mixing band alpha mode exists to avoid.
  static constexpr float kMaxFeatherFraction = 1.0f;
  /// Feather width as a fraction of the narrowest camera footprint, matching the reference
  /// implementation's `blend_width`.
  static constexpr float kDefaultFeatherFraction = 0.05f;

  // Implicit by design: `Stitcher(batch, num_levels, masks, ...)` keeps compiling and keeps its
  // exact meaning.
  // NOLINTNEXTLINE(google-explicit-constructor)
  constexpr BlendSettings(int levels)
      : mode(levels > 0 ? BlendMode::kLaplacian : BlendMode::kHardSeam), num_levels(levels) {}

  static constexpr BlendSettings HardSeam() {
    return BlendSettings(0);
  }

  /// Unlike the bare-int constructor, this always means Laplacian: a level count below one is an
  /// error from Validate() rather than a silent hard seam.
  static constexpr BlendSettings Laplacian(int levels) {
    BlendSettings settings(levels);
    settings.mode = BlendMode::kLaplacian;
    settings.num_levels = levels;
    return settings;
  }

  /// Requires a floating-point stitcher compute type; integer pipeline/input pixels remain supported.
  /// @param feather_fraction Crossfade width as a fraction of the narrowest camera footprint width.
  ///        0 degenerates to a hard seam.
  static constexpr BlendSettings Alpha(float feather_fraction = kDefaultFeatherFraction) {
    BlendSettings settings(0);
    settings.mode = BlendMode::kAlpha;
    settings.feather_fraction = feather_fraction;
    return settings;
  }

  BlendMode mode;
  int num_levels; ///< Meaningful only when mode == kLaplacian. Negative values are rejected.
  float feather_fraction{0.0f}; ///< Meaningful only when mode == kAlpha.

  constexpr bool is_hard_seam() const {
    return mode == BlendMode::kHardSeam;
  }
  /// True when the operator mixes contributors, i.e. it needs the soft-seam remap and blend path.
  constexpr bool is_soft() const {
    return mode != BlendMode::kHardSeam;
  }

  /// Number of pyramid levels the blend ROI must account for. Alpha and hard seam are pointwise.
  constexpr int roi_levels() const {
    return mode == BlendMode::kLaplacian ? num_levels : 0;
  }

  /// Returns an error message when the settings are unusable, or an empty string when they are fine.
  std::string Validate() const;
};

} // namespace pano
} // namespace hm
