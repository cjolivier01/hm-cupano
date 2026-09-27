#include "cupano/pano/blendMode.h"

#include <cmath>

namespace hm {
namespace pano {

std::string BlendSettings::Validate() const {
  switch (mode) {
    case BlendMode::kHardSeam:
      // A bare int reaches here for 0 and for negatives. Zero is the historical hard-seam spelling;
      // a negative level count is a caller mistake, not a mode.
      if (num_levels < 0) {
        return "Blend level count must not be negative";
      }
      return {};
    case BlendMode::kLaplacian:
      if (num_levels < 1) {
        return "Laplacian blending requires at least one pyramid level";
      }
      return {};
    case BlendMode::kAlpha:
      if (!std::isfinite(feather_fraction)) {
        return "Alpha blend feather fraction must be finite";
      }
      if (feather_fraction < 0.0f || feather_fraction > kMaxFeatherFraction) {
        return "Alpha blend feather fraction must be in [0, 1]";
      }
      return {};
  }
  return "Unknown blend mode";
}

} // namespace pano
} // namespace hm
