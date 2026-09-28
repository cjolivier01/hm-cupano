// blendMode_test.cpp

#include <gtest/gtest.h>

#include <limits>

#include "cupano/pano/blendMode.h"

namespace {

using hm::pano::BlendMode;
using hm::pano::BlendSettings;

// A bare int keeps the historical encoding: 0 means a hard seam, anything above means Laplacian.
// Every existing caller relies on this, including hstream across the Bazel pin.
TEST(BlendSettingsTest, BareIntPreservesTheHistoricalEncoding) {
  const BlendSettings hard = 0;
  EXPECT_EQ(hard.mode, BlendMode::kHardSeam);
  EXPECT_TRUE(hard.is_hard_seam());
  EXPECT_FALSE(hard.is_soft());
  EXPECT_EQ(hard.roi_levels(), 0);
  EXPECT_TRUE(hard.Validate().empty());

  for (int levels : {1, 2, 6, 11}) {
    const BlendSettings laplacian = levels;
    EXPECT_EQ(laplacian.mode, BlendMode::kLaplacian) << levels;
    EXPECT_FALSE(laplacian.is_hard_seam()) << levels;
    EXPECT_TRUE(laplacian.is_soft()) << levels;
    EXPECT_EQ(laplacian.num_levels, levels);
    EXPECT_EQ(laplacian.roi_levels(), levels);
    EXPECT_TRUE(laplacian.Validate().empty()) << levels;
  }
}

// A negative count is a caller mistake, not the hard-seam spelling.
TEST(BlendSettingsTest, NegativeLevelCountIsRejected) {
  const BlendSettings negative = -3;
  EXPECT_FALSE(negative.Validate().empty());
}

// Unlike the bare-int constructor, the factory always means Laplacian, so a level count below one
// is an error rather than a silent hard seam.
TEST(BlendSettingsTest, LaplacianFactoryRejectsTooFewLevels) {
  EXPECT_TRUE(BlendSettings::Laplacian(1).Validate().empty());
  EXPECT_EQ(BlendSettings::Laplacian(4).mode, BlendMode::kLaplacian);

  const BlendSettings zero = BlendSettings::Laplacian(0);
  EXPECT_EQ(zero.mode, BlendMode::kLaplacian) << "Laplacian(0) must not silently become a hard seam";
  EXPECT_FALSE(zero.Validate().empty());
  EXPECT_FALSE(BlendSettings::Laplacian(-1).Validate().empty());
}

TEST(BlendSettingsTest, AlphaValidatesItsFraction) {
  EXPECT_TRUE(BlendSettings::Alpha(0.0f).Validate().empty());
  EXPECT_TRUE(BlendSettings::Alpha(BlendSettings::kDefaultFeatherFraction).Validate().empty());
  EXPECT_TRUE(BlendSettings::Alpha(BlendSettings::kMaxFeatherFraction).Validate().empty());

  EXPECT_FALSE(BlendSettings::Alpha(-0.01f).Validate().empty());
  EXPECT_FALSE(BlendSettings::Alpha(BlendSettings::kMaxFeatherFraction + 0.01f).Validate().empty());
  EXPECT_FALSE(BlendSettings::Alpha(std::numeric_limits<float>::quiet_NaN()).Validate().empty());

  // Alpha is a soft mode with no pyramid, so the blend ROI sees zero levels.
  const BlendSettings alpha = BlendSettings::Alpha(0.1f);
  EXPECT_TRUE(alpha.is_soft());
  EXPECT_FALSE(alpha.is_hard_seam());
  EXPECT_EQ(alpha.roi_levels(), 0);
}

TEST(BlendSettingsTest, HardSeamFactoryMatchesTheBareZero) {
  const BlendSettings factory = BlendSettings::HardSeam();
  const BlendSettings bare = 0;
  EXPECT_EQ(factory.mode, bare.mode);
  EXPECT_EQ(factory.num_levels, bare.num_levels);
  EXPECT_EQ(factory.roi_levels(), bare.roi_levels());
  EXPECT_TRUE(factory.Validate().empty());
}

} // namespace
