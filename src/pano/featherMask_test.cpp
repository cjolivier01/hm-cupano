// featherMask_test.cpp

#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include <cmath>
#include <vector>

#include "cupano/pano/featherMask.h"

namespace {

using hm::pano::feather::build_weights;
using hm::pano::feather::coverage_masks;
using hm::pano::feather::Params;
using hm::pano::feather::Result;

constexpr uint16_t kUnmapped = 65535;

float smoothstep(float t) {
  t = std::min(1.0f, std::max(0.0f, t));
  return t * t * (3.0f - 2.0f * t);
}

// A remap that maps every pixel of a w x h footprint.
cv::Mat full_map(int w, int h, uint16_t value = 0) {
  return cv::Mat(h, w, CV_16U, cv::Scalar(value));
}

struct Rig {
  std::vector<cv::Mat> remap_x;
  std::vector<cv::Mat> remap_y;
  std::vector<cv::Point> positions;
};

Rig make_rig(const std::vector<cv::Size>& sizes, const std::vector<cv::Point>& positions) {
  Rig rig;
  for (const auto& s : sizes) {
    rig.remap_x.push_back(full_map(s.width, s.height));
    rig.remap_y.push_back(full_map(s.width, s.height));
  }
  rig.positions = positions;
  return rig;
}

// Vertical seam: label 0 for x < x0, label 1 for x >= x0. Both cameras cover the whole canvas.
cv::Mat vertical_seam(int w, int h, int x0) {
  cv::Mat labels(h, w, CV_8U, cv::Scalar(0));
  labels.colRange(x0, w).setTo(1);
  return labels;
}

float weight_at(const cv::Mat& weights, int x, int y, int channel) {
  const int n = weights.channels();
  return weights.ptr<float>(y)[x * n + channel];
}

float sum_at(const cv::Mat& weights, int x, int y) {
  const int n = weights.channels();
  const float* p = weights.ptr<float>(y) + x * n;
  float s = 0.0f;
  for (int c = 0; c < n; ++c) {
    s += p[c];
  }
  return s;
}

} // namespace

// The signed-distance ramp must reproduce a smoothstep centred exactly on the geometric boundary
// between the two pixel columns that straddle the seam.
TEST(FeatherMaskTest, TwoCameraProfileMatchesClosedForm) {
  constexpr int W = 256, H = 16, x0 = 128;
  constexpr float R = 32.0f;
  const Rig rig = make_rig({{W, H}, {W, H}}, {{0, 0}, {0, 0}});
  const cv::Mat labels = vertical_seam(W, H, x0);

  Params params;
  params.fraction = R / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  ASSERT_FALSE(r.hard);
  EXPECT_NEAR(r.radius_px, R, 1e-3f);
  EXPECT_FALSE(r.overlap_capped);

  const int y = H / 2;
  for (int x = 0; x < W; ++x) {
    const float s = static_cast<float>(x) - static_cast<float>(x0) + 0.5f;
    const float expected = smoothstep(0.5f - s / R);
    EXPECT_NEAR(weight_at(r.weights, x, y, 0), expected, 1e-5f) << "x=" << x;
    EXPECT_NEAR(weight_at(r.weights, x, y, 1), 1.0f - expected, 1e-5f) << "x=" << x;
  }
}

// S(t) + S(1-t) == 1, so two cameras across a seam need no normalization at all.
// Weights are normalized here rather than in the kernels, because only the N kernel normalizes
// unconditionally. Every covered pixel must sum to exactly one for all camera counts.
TEST(FeatherMaskTest, WeightsSumToOneForEveryCameraCount) {
  constexpr int W = 240, H = 40;
  for (int n : {2, 3, 5, 8}) {
    std::vector<cv::Size> sizes(n, cv::Size(W, H));
    std::vector<cv::Point> positions(n, cv::Point(0, 0));
    const Rig rig = make_rig(sizes, positions);
    cv::Mat labels(H, W, CV_8U);
    for (int x = 0; x < W; ++x)
      labels.col(x).setTo(std::min(n - 1, x * n / W));
    Params params;
    params.fraction = 16.0f / static_cast<float>(W);
    const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, n, params);
    ASSERT_TRUE(r.error.empty()) << r.error;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        EXPECT_NEAR(sum_at(r.weights, x, y), 1.0f, 1e-6f) << "n=" << n << " (" << x << "," << y << ")";
      }
    }
  }
}

TEST(FeatherMaskTest, TwoCameraWeightsSumToOne) {
  constexpr int W = 200, H = 24, x0 = 90;
  const Rig rig = make_rig({{W, H}, {W, H}}, {{0, 0}, {0, 0}});
  const cv::Mat labels = vertical_seam(W, H, x0);

  for (float fraction : {0.01f, 0.05f, 0.2f, 0.5f}) {
    Params params;
    params.fraction = fraction;
    const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
    ASSERT_TRUE(r.error.empty()) << r.error;
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        EXPECT_NEAR(sum_at(r.weights, x, y), 1.0f, 1e-6f) << "fraction=" << fraction << " (" << x << "," << y << ")";
      }
    }
  }
}

// The ramp must saturate to exactly 1 and exactly 0 outside the band, so the vast majority of the
// canvas carries unmodified source pixels.
TEST(FeatherMaskTest, SaturatesOutsideTheBand) {
  constexpr int W = 256, H = 8, x0 = 128;
  constexpr float R = 32.0f;
  const Rig rig = make_rig({{W, H}, {W, H}}, {{0, 0}, {0, 0}});
  const cv::Mat labels = vertical_seam(W, H, x0);

  Params params;
  params.fraction = R / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  const int y = H / 2;
  for (int x = 0; x < W; ++x) {
    const float s = static_cast<float>(x) - static_cast<float>(x0) + 0.5f;
    if (s <= -R / 2.0f) {
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 0), 1.0f) << "x=" << x;
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 1), 0.0f) << "x=" << x;
    } else if (s >= R / 2.0f) {
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 0), 0.0f) << "x=" << x;
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 1), 1.0f) << "x=" << x;
    }
  }
}

// A hard edge anywhere in the field is the failure this feature exists to avoid. The raw ramp's
// analytic maximum slope is 1.5/R per pixel; host-side normalization near a multi-region junction
// amplifies that to a measured 1.13x, so the bound here is 3/R.
//
// The fixture has uniform full coverage on purpose, so the per-pixel cap does not engage and the
// local radius is constant. The field is 3/R_local-Lipschitz, not 3/radius_px, so this bound does
// not generalize to geometries where the cap varies: a camera with no data at all must reach zero
// weight over however many pixels the hole spans, exactly as the hard-seam path does.
TEST(FeatherMaskTest, GradientIsBounded) {
  constexpr int W = 160, H = 120;
  constexpr float R = 24.0f;
  const Rig rig = make_rig({{W, H}, {W, H}, {W, H}}, {{0, 0}, {0, 0}, {0, 0}});

  // Three regions meeting at a point, so the check covers a triple junction too.
  cv::Mat labels(H, W, CV_8U, cv::Scalar(0));
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (y < H / 2) {
        labels.at<uint8_t>(y, x) = (x < W / 2) ? 0 : 1;
      } else {
        labels.at<uint8_t>(y, x) = 2;
      }
    }
  }

  Params params;
  params.fraction = R / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 3, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  const float limit = 3.0f / r.radius_px + 1e-5f;
  for (int c = 0; c < 3; ++c) {
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float w = weight_at(r.weights, x, y, c);
        if (x + 1 < W) {
          EXPECT_LE(std::abs(weight_at(r.weights, x + 1, y, c) - w), limit)
              << "ch " << c << " at (" << x << "," << y << ")";
        }
        if (y + 1 < H) {
          EXPECT_LE(std::abs(weight_at(r.weights, x, y + 1, c) - w), limit)
              << "ch " << c << " at (" << x << "," << y << ")";
        }
      }
    }
  }
}

// Three regions meeting at a point must produce an equal share, not a dark spot or a discontinuity.
TEST(FeatherMaskTest, TriplePointSplitsEvenly) {
  constexpr int W = 161, H = 161;
  const Rig rig = make_rig({{W, H}, {W, H}, {W, H}}, {{0, 0}, {0, 0}, {0, 0}});
  const int cx = W / 2, cy = H / 2;

  // Three 120-degree sectors about the centre.
  cv::Mat labels(H, W, CV_8U, cv::Scalar(0));
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const double angle = std::atan2(static_cast<double>(y - cy), static_cast<double>(x - cx));
      const double turns = (angle + CV_PI) / (2.0 * CV_PI);
      labels.at<uint8_t>(y, x) = static_cast<uint8_t>(std::min(2, static_cast<int>(turns * 3.0)));
    }
  }

  Params params;
  params.fraction = 32.0f / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 3, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  // The field arrives normalized, so this sum is one and the division below is a no-op; it is
  // written as a ratio so the even-split check does not silently also depend on that.
  const float total = sum_at(r.weights, cx, cy);
  ASSERT_GT(total, 0.0f);
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(weight_at(r.weights, cx, cy, c) / total, 1.0f / 3.0f, 0.05f) << "channel " << c;
  }

  // Three overlapping regions are exactly where the raw smoothstep weights exceed one (the
  // S(t)+S(1-t)==1 identity only covers two). BatchedBlendKernel3 does not normalize for
  // three-channel compute and the two-image kernel never does, so the field has to arrive
  // normalized. Without it the apex sums to about 1.5 and renders 50% too bright.
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      EXPECT_NEAR(sum_at(r.weights, x, y), 1.0f, 1e-6f) << "(" << x << "," << y << ")";
    }
  }
}

// A camera must contribute nothing outside its own remap footprint. The blend kernels only drop
// zero-alpha contributors when CHANNELS == 4, so on a three-channel compute type this property is
// the only thing preventing the memset black outside a footprint from darkening the seam.
TEST(FeatherMaskTest, WeightsAreZeroOutsideCoverage) {
  constexpr int W = 200, H = 32;
  // Camera 0 covers [0,120), camera 1 covers [80,200). Seam at x=100, inside the overlap.
  const Rig rig = make_rig({{120, H}, {120, H}}, {{0, 0}, {80, 0}});
  const cv::Mat labels = vertical_seam(W, H, 100);

  Params params;
  params.fraction = 16.0f / 120.0f;
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (x >= 120) {
        EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 0), 0.0f) << "camera 0 past its footprint at x=" << x;
      }
      if (x < 80) {
        EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 1), 0.0f) << "camera 1 before its footprint at x=" << x;
      }
      EXPECT_GT(sum_at(r.weights, x, y), 0.0f) << "(" << x << "," << y << ")";
    }
  }
}

// The radius must shrink to fit a narrow overlap rather than ramping toward a camera with no data.
TEST(FeatherMaskTest, RadiusIsCappedByOverlap) {
  constexpr int W = 200, H = 16;
  // Overlap is only [90,110): 20 pixels wide. Seam at its centre.
  const Rig rig = make_rig({{110, H}, {110, H}}, {{0, 0}, {90, 0}});
  const cv::Mat labels = vertical_seam(W, H, 100);

  Params params;
  params.fraction = 0.9f; // ~99 px, far wider than the overlap supports
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  EXPECT_TRUE(r.overlap_capped);
  EXPECT_FALSE(r.hard);
  // Exactly twice the shallowest coverage at the seam, not merely in the right ballpark: a loose
  // bound here cannot tell 2*coverage from 1*coverage, and the factor of two is the whole cap.
  EXPECT_FLOAT_EQ(r.radius_px, 20.0f);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      EXPECT_GT(sum_at(r.weights, x, y), 0.0f) << "(" << x << "," << y << ")";
    }
  }
}

// The cap has to be over the cameras that actually contribute at a pixel, not over "covered by at
// least two cameras". A third camera blanketing the area inflates that count, so the cap would
// never engage while the two cameras meeting at the seam are strangled by their own tapers - the
// caller asks for a 40 px feather and silently gets a hard seam.
TEST(FeatherMaskTest, BlanketingThirdCameraDoesNotDefeatTheCap) {
  constexpr int W = 300, H = 60, aw = 150, ab_overlap = 10;
  constexpr int bx = aw - ab_overlap;
  Params params;
  params.fraction = 40.0f / static_cast<float>(aw);

  // Control: just the two cameras that meet at the seam.
  Rig pair;
  pair.remap_x = {full_map(aw, H), full_map(W - bx, H)};
  pair.remap_y = {full_map(aw, H), full_map(W - bx, H)};
  pair.positions = {{0, 0}, {bx, 0}};
  cv::Mat two_labels(H, W, CV_8U, cv::Scalar(0));
  two_labels.colRange(aw - ab_overlap / 2, W).setTo(1);
  const Result control = build_weights(two_labels, pair.remap_x, pair.remap_y, pair.positions, 2, params);
  ASSERT_TRUE(control.error.empty()) << control.error;
  EXPECT_TRUE(control.overlap_capped);
  EXPECT_LE(control.min_seam_radius_px, static_cast<float>(ab_overlap));

  // Same seam, plus a third camera covering the whole canvas and owning a sliver.
  Rig trio;
  trio.remap_x = {full_map(aw, H), full_map(W - bx, H), full_map(W, H)};
  trio.remap_y = {full_map(aw, H), full_map(W - bx, H), full_map(W, H)};
  trio.positions = {{0, 0}, {bx, 0}, {0, 0}};
  cv::Mat three_labels = two_labels.clone();
  three_labels.rowRange(H - 5, H).setTo(2);
  const Result blanketed = build_weights(three_labels, trio.remap_x, trio.remap_y, trio.positions, 3, params);
  ASSERT_TRUE(blanketed.error.empty()) << blanketed.error;
  EXPECT_TRUE(blanketed.overlap_capped) << "the blanketing camera hid the narrow A/B overlap";
  EXPECT_LE(blanketed.min_seam_radius_px, static_cast<float>(ab_overlap))
      << "the A/B seam must still be capped at its own overlap";
}

// The mirror image of the blanketing case: a camera whose region sits near another seam but which
// covers nothing there must not be treated as a contributor. Its weight is exactly zero (its taper
// reads coverage_distance == 0), so letting it into the cap's min would drive the radius to zero
// and turn a perfectly healthy seam hard.
TEST(FeatherMaskTest, NonCoveringNeighbourDoesNotCollapseTheCap) {
  constexpr int W = 600, H = 200, seam_x = 250;
  // fraction * narrowest = 200, so the contributor reach is 100.5 px - wide enough to see the
  // island 50 px away from the 0/1 seam.
  Params params;
  params.fraction = 0.5f;

  Rig rig;
  rig.remap_x = {full_map(400, H), full_map(450, H)};
  rig.remap_y = {full_map(400, H), full_map(450, H)};
  rig.positions = {{0, 0}, {150, 0}};
  cv::Mat two_labels(H, W, CV_8U, cv::Scalar(0));
  two_labels.colRange(seam_x, W).setTo(1);
  const Result control = build_weights(two_labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(control.error.empty()) << control.error;
  ASSERT_FALSE(control.hard);
  ASSERT_GT(control.radius_px, 100.0f);

  // Camera 2 owns an island 50 px from the 0/1 seam but covers nothing there. Its remap is as wide
  // as camera 0 so `narrowest`, and therefore the requested radius, is unchanged; only the mapped
  // region is small.
  Rig with_island = rig;
  cv::Mat island_x(H, 400, CV_16U, cv::Scalar(kUnmapped));
  cv::Mat island_y(H, 400, CV_16U, cv::Scalar(kUnmapped));
  island_x(cv::Rect(0, 80, 60, 40)).setTo(0);
  island_y(cv::Rect(0, 80, 60, 40)).setTo(0);
  with_island.remap_x.push_back(island_x);
  with_island.remap_y.push_back(island_y);
  with_island.positions.push_back({300, 0});
  cv::Mat three_labels = two_labels.clone();
  three_labels(cv::Rect(300, 80, 60, 40)).setTo(2);

  const Result islanded =
      build_weights(three_labels, with_island.remap_x, with_island.remap_y, with_island.positions, 3, params);
  ASSERT_TRUE(islanded.error.empty()) << islanded.error;
  EXPECT_FALSE(islanded.hard) << "a non-covering neighbour collapsed the whole field";
  EXPECT_GE(islanded.radius_px, control.radius_px * 0.99f) << "a non-covering neighbour narrowed the 0/1 crossfade";
}

// Counts pixels along a row whose weight for `channel` is strictly between the two rails, which
// is the crossfade's realised width there.
int ramp_width(const cv::Mat& weights, int y, int x_lo, int x_hi, int channel) {
  int n = 0;
  for (int x = x_lo; x < x_hi; ++x) {
    const float w = weight_at(weights, x, y, channel);
    if (w > 0.02f && w < 0.98f) {
      ++n;
    }
  }
  return n;
}

// A camera whose weight at p is exactly zero must not pin the cap there. Testing reach against the
// requested radius made it pin anyway: camera 2 below sits 60 px from the 0/1 seam, so once the
// band narrows it cannot reach, but the shallow footprint it drags across the seam was enough to
// collapse a 400 px crossfade to 4 px.
TEST(FeatherMaskTest, GrazingNeighbourWithZeroWeightDoesNotCollapseTheCap) {
  constexpr int W = 600, H = 400, seam_x = 250;
  constexpr int sliver_y = 198, sliver_h = 5, island_x = 310, island_w = 20;
  Params params;
  params.fraction = 1.0f; // 1.0 * 400 = a 400 px request, far wider than the grazing footprint.

  Rig rig;
  rig.remap_x = {full_map(400, H), full_map(400, H)};
  rig.remap_y = {full_map(400, H), full_map(400, H)};
  rig.positions = {{0, 0}, {150, 0}};
  const cv::Mat two_labels = vertical_seam(W, H, seam_x);
  const Result control = build_weights(two_labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(control.error.empty()) << control.error;
  const int control_ramp = ramp_width(control.weights, sliver_y + 2, 180, 290, 0);
  ASSERT_GT(control_ramp, 40) << "fixture is wrong: the two-camera seam should feather widely";

  // Camera 2 owns a 20 px island 60 px to the right of the 0/1 seam, and drags a 5 px tall
  // footprint back across it. Its remap is as wide as the others so the request is unchanged.
  Rig grazing = rig;
  cv::Mat sliver_x(H, 400, CV_16U, cv::Scalar(kUnmapped));
  cv::Mat sliver_y_map(H, 400, CV_16U, cv::Scalar(kUnmapped));
  // Footprint x in [198, 332) on the canvas, so it straddles the seam at 250.
  sliver_x(cv::Rect(0, sliver_y, 134, sliver_h)).setTo(0);
  sliver_y_map(cv::Rect(0, sliver_y, 134, sliver_h)).setTo(0);
  grazing.remap_x.push_back(sliver_x);
  grazing.remap_y.push_back(sliver_y_map);
  grazing.positions.push_back({198, 0});
  cv::Mat three_labels = two_labels.clone();
  three_labels(cv::Rect(island_x, sliver_y, island_w, sliver_h)).setTo(2);

  const Result grazed = build_weights(three_labels, grazing.remap_x, grazing.remap_y, grazing.positions, 3, params);
  ASSERT_TRUE(grazed.error.empty()) << grazed.error;
  ASSERT_FALSE(grazed.hard);

  // Camera 2 has zero weight mathematically at the cutoff. OpenCV's vectorized division can
  // leave a tiny positive residual before smoothstep on ARM (about 1e-17 after normalization).
  // Allow float rounding here; the ramp-width assertion below pins the cap's actual behavior.
  EXPECT_NEAR(weight_at(grazed.weights, seam_x, sliver_y + 2, 2), 0.0f, 1e-7f);
  const int grazed_ramp = ramp_width(grazed.weights, sliver_y + 2, 180, 290, 0);
  EXPECT_GE(grazed_ramp, control_ramp / 2)
      << "a grazing camera with zero weight collapsed the crossfade: " << grazed_ramp << " px against " << control_ramp;
}

// The ROI pad is sized from the request, not from the widest seam pixel: where the cap bites, the
// ramp still runs out to the full requested radius elsewhere under the same ROI.
TEST(FeatherMaskTest, RequestedRadiusReportsTheRequestEvenWhenTheCapBites) {
  constexpr int W = 600, H = 400, seam_x = 396;
  Params params;
  params.fraction = 1.0f; // 400 px requested against an 8 px overlap.

  Rig rig;
  // An 8 px overlap, so the cap bites along the *whole* seam rather than at one pinch. Without
  // that, radius_px happens to equal the request and the two cannot be told apart.
  rig.remap_x = {full_map(400, H), full_map(400, H)};
  rig.remap_y = {full_map(400, H), full_map(400, H)};
  rig.positions = {{0, 0}, {392, 0}};
  const cv::Mat labels = vertical_seam(W, H, seam_x);

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  ASSERT_FALSE(r.hard);
  EXPECT_FLOAT_EQ(r.requested_radius_px, 400.0f) << "the request must survive the cap";
  EXPECT_LT(r.radius_px, 16.0f) << "the fixture must actually be capped everywhere";
  EXPECT_FLOAT_EQ(r.min_seam_radius_px, r.radius_px);
  EXPECT_TRUE(r.overlap_capped);
  // Every seam pixel is capped, which also pins the share's denominator to the seam rather than
  // to the canvas.
  EXPECT_FLOAT_EQ(r.capped_seam_fraction, 1.0f);
}

// min_seam_radius_px is a single worst pixel, and on a real rig it is almost always tiny. Reporting
// only that trains operators to ignore the message, so the share of the seam that was actually
// narrowed is reported alongside it.
TEST(FeatherMaskTest, CappedShareSeparatesALocalPinchFromAGeneralOne) {
  constexpr int W = 600, H = 400, seam_x = 250;
  Params params;
  params.fraction = 0.1f; // 40 px, comfortably inside the overlap along most of the seam.

  Rig rig;
  rig.remap_x = {full_map(400, H), full_map(400, H)};
  rig.remap_y = {full_map(400, H), full_map(400, H)};
  rig.positions = {{0, 0}, {150, 0}};
  cv::Mat labels = vertical_seam(W, H, seam_x);
  // A small island handed to camera 1 where camera 1 has only a couple of pixels of depth.
  labels(cv::Rect(152, 0, 8, 4)).setTo(1);

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  EXPECT_TRUE(r.overlap_capped);
  EXPECT_LT(r.min_seam_radius_px, 8.0f) << "the island should pinch hard";
  EXPECT_FLOAT_EQ(r.radius_px, 40.0f) << "the rest of the seam should be untouched";
  EXPECT_GT(r.capped_seam_fraction, 0.0f);
  EXPECT_LT(r.capped_seam_fraction, 0.2f) << "a local pinch must not read as a general one";
}

// The seam label can name a camera that does not cover the pixel. Correcting the labels to a
// covering camera is what keeps the partition total non-zero there; without it the normalization
// guard leaves the pixel black.
TEST(FeatherMaskTest, CoverageCorrectedLabelsKeepThePartitionTotal) {
  constexpr int W = 200, H = 120, seam_x = 100;
  Params params;
  params.fraction = 0.2f;

  Rig rig;
  // Camera 0's footprint stops at x = 60, well short of the region the seam hands it.
  cv::Mat narrow_x(H, 200, CV_16U, cv::Scalar(kUnmapped));
  cv::Mat narrow_y(H, 200, CV_16U, cv::Scalar(kUnmapped));
  narrow_x(cv::Rect(0, 0, 60, H)).setTo(0);
  narrow_y(cv::Rect(0, 0, 60, H)).setTo(0);
  rig.remap_x = {narrow_x, full_map(200, H)};
  rig.remap_y = {narrow_y, full_map(200, H)};
  rig.positions = {{0, 0}, {0, 0}};
  const cv::Mat labels = vertical_seam(W, H, seam_x);

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  const std::vector<cv::Mat> coverage = coverage_masks({W, H}, rig.remap_x, rig.remap_y, rig.positions);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const bool any_covers = coverage[0].at<uint8_t>(y, x) != 0 || coverage[1].at<uint8_t>(y, x) != 0;
      if (any_covers) {
        ASSERT_NEAR(sum_at(r.weights, x, y), 1.0f, 1e-4f) << "black pixel at " << x << "," << y;
      }
    }
  }
}

// The reported statistics are taken over both sides of every label change. Sampling only the left
// side hides a camera whose footprint starts on the right flank of the seam.
TEST(FeatherMaskTest, SeamStatisticsCoverBothSidesOfALabelChange) {
  constexpr int W = 400, H = 200, seam_x = 200;
  Params params;
  params.fraction = 0.5f;

  Rig rig;
  // Camera 1's footprint begins exactly at the seam, so its coverage depth is zero on the right
  // flank and one pixel deep nowhere to the left of it.
  cv::Mat right_x(H, 400, CV_16U, cv::Scalar(kUnmapped));
  cv::Mat right_y(H, 400, CV_16U, cv::Scalar(kUnmapped));
  right_x(cv::Rect(seam_x, 0, W - seam_x, H)).setTo(0);
  right_y(cv::Rect(seam_x, 0, W - seam_x, H)).setTo(0);
  rig.remap_x = {full_map(400, H), right_x};
  rig.remap_y = {full_map(400, H), right_y};
  rig.positions = {{0, 0}, {0, 0}};
  const cv::Mat labels = vertical_seam(W, H, seam_x);

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  EXPECT_TRUE(r.overlap_capped) << "the right flank of the seam has no coverage depth to feather into";
  EXPECT_LT(r.min_seam_radius_px, 8.0f);
}

// The hard fallback builds its one-hot field from the labels handed in, not the corrected ones.
// The point of that path is to match the hard-seam mask byte for byte, and the correction would
// move a seam the hard-seam mask does not have.
TEST(FeatherMaskTest, HardFallbackKeepsTheUncorrectedLabels) {
  constexpr int W = 128, H = 40, seam_x = 60;
  Params params;
  params.fraction = 0.0f; // Forces the hard fallback.

  Rig rig;
  cv::Mat holed_x = full_map(W, H);
  cv::Mat holed_y = full_map(W, H);
  // A hole inside camera 0's own region that camera 1 covers, so the correction has somewhere to
  // reassign to and the two label images genuinely differ.
  holed_x(cv::Rect(20, 8, 30, 20)).setTo(kUnmapped);
  holed_y(cv::Rect(20, 8, 30, 20)).setTo(kUnmapped);
  rig.remap_x = {holed_x, full_map(W, H)};
  rig.remap_y = {holed_y, full_map(W, H)};
  rig.positions = {{0, 0}, {0, 0}};
  const cv::Mat labels = vertical_seam(W, H, seam_x);

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  ASSERT_TRUE(r.hard) << "zero width must take the hard fallback";

  // The fixture only means anything if the correction would have moved something.
  const std::vector<cv::Mat> coverage = coverage_masks({W, H}, rig.remap_x, rig.remap_y, rig.positions);
  int would_move = 0;
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (labels.at<uint8_t>(y, x) == 0 && !coverage[0].at<uint8_t>(y, x) && coverage[1].at<uint8_t>(y, x))
        ++would_move;
    }
  }
  ASSERT_GT(would_move, 0) << "fixture does not exercise the correction";

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      const int owner = labels.at<uint8_t>(y, x);
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, owner), 1.0f) << x << "," << y;
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 1 - owner), 0.0f) << x << "," << y;
      EXPECT_EQ(r.corrected_labels.at<uint8_t>(y, x), owner) << x << "," << y;
    }
  }
}

// The correction reassigns to the camera covering the pixel most deeply, not to the first one
// that covers it. The deepest coverer has the most room for its taper, so picking any other
// leaves the reassigned pixel closer to a footprint edge than it needed to be.
TEST(FeatherMaskTest, CorrectionPicksTheDeepestCoverer) {
  constexpr int W = 200, H = 60, hole_x = 40, hole_w = 30;
  Params params;
  params.fraction = 0.1f;

  Rig rig;
  // Camera 0 owns everything but loses a block. Camera 1 grazes that block with a shallow strip;
  // camera 2 covers it deeply. Camera 1 comes first, so "first coverer" and "deepest coverer"
  // disagree.
  cv::Mat owner_x = full_map(W, H), owner_y = full_map(W, H);
  owner_x(cv::Rect(hole_x, 10, hole_w, 20)).setTo(kUnmapped);
  owner_y(cv::Rect(hole_x, 10, hole_w, 20)).setTo(kUnmapped);
  cv::Mat shallow_x(H, W, CV_16U, cv::Scalar(kUnmapped)), shallow_y(H, W, CV_16U, cv::Scalar(kUnmapped));
  shallow_x(cv::Rect(hole_x, 10, hole_w, 4)).setTo(0);
  shallow_y(cv::Rect(hole_x, 10, hole_w, 4)).setTo(0);
  cv::Mat deep_x(H, W, CV_16U, cv::Scalar(kUnmapped)), deep_y(H, W, CV_16U, cv::Scalar(kUnmapped));
  deep_x(cv::Rect(hole_x - 20, 0, hole_w + 40, 40)).setTo(0);
  deep_y(cv::Rect(hole_x - 20, 0, hole_w + 40, 40)).setTo(0);
  rig.remap_x = {owner_x, shallow_x, deep_x};
  rig.remap_y = {owner_y, shallow_y, deep_y};
  rig.positions = {{0, 0}, {0, 0}, {0, 0}};
  const cv::Mat labels(H, W, CV_8U, cv::Scalar(0));

  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 3, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  ASSERT_FALSE(r.corrected_labels.empty());
  // Inside the overlap of the shallow and deep coverers, the deep one must win.
  for (int y = 10; y < 14; ++y) {
    for (int x = hole_x; x < hole_x + hole_w; ++x) {
      EXPECT_EQ(r.corrected_labels.at<uint8_t>(y, x), 2) << x << "," << y;
    }
  }
}

// Widths below one pixel preserve the hard-seam mask and all hard-result metadata.
TEST(FeatherMaskTest, SubpixelWidthYieldsExactOneHot) {
  constexpr int W = 64, H = 16, x0 = 30;
  const Rig rig = make_rig({{W, H}, {W, H}}, {{0, 0}, {0, 0}});
  const cv::Mat labels = vertical_seam(W, H, x0);

  for (const Params& params : {Params{0.0f}, Params{0.5f / W}, Params{0.05f, 0.0f}, Params{0.05f, 0.5f}}) {
    const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
    ASSERT_TRUE(r.error.empty()) << r.error;
    EXPECT_TRUE(r.hard);
    EXPECT_FLOAT_EQ(r.radius_px, 0.0f);
    EXPECT_FLOAT_EQ(r.min_seam_radius_px, 0.0f);
    EXPECT_FLOAT_EQ(r.requested_radius_px, 0.0f);
    EXPECT_FLOAT_EQ(r.capped_seam_fraction, 0.0f);
    EXPECT_FALSE(r.overlap_capped);
    EXPECT_EQ(r.corrected_labels.data, labels.data);

    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const float expected0 = (x < x0) ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 0), expected0);
        EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 1), 1.0f - expected0);
      }
    }
  }
}

// distanceTransform treats outside-the-image as foreground, so the canvas border must not behave
// like a seam. A future BORDER_CONSTANT-style change would break this.
TEST(FeatherMaskTest, CanvasBorderDoesNotFeather) {
  constexpr int H = 32, pad = 40;
  constexpr int W = 200;
  const int x0 = 100;
  const Rig small = make_rig({{W, H}, {W, H}}, {{0, 0}, {0, 0}});
  const cv::Mat labels_small = vertical_seam(W, H, x0);

  const int W2 = W + 2 * pad;
  const Rig big = make_rig({{W2, H}, {W2, H}}, {{0, 0}, {0, 0}});
  cv::Mat labels_big(H, W2, CV_8U, cv::Scalar(0));
  labels_big.colRange(x0 + pad, W2).setTo(1);

  Params params;
  params.fraction = 32.0f / static_cast<float>(W);
  const Result a = build_weights(labels_small, small.remap_x, small.remap_y, small.positions, 2, params);
  ASSERT_TRUE(a.error.empty()) << a.error;
  Params params_big;
  params_big.fraction = a.radius_px / static_cast<float>(W2);
  const Result b = build_weights(labels_big, big.remap_x, big.remap_y, big.positions, 2, params_big);
  ASSERT_TRUE(b.error.empty()) << b.error;
  ASSERT_NEAR(a.radius_px, b.radius_px, 1e-3f);

  const int y = H / 2;
  for (int x = 0; x < W; ++x) {
    EXPECT_NEAR(weight_at(a.weights, x, y, 0), weight_at(b.weights, x + pad, y, 0), 1e-5f) << "x=" << x;
  }
}

// Eight cameras is the supported maximum; the field must stay finite and in range.
TEST(FeatherMaskTest, EightCamerasStayInRange) {
  constexpr int W = 400, H = 24, n = 8;
  std::vector<cv::Size> sizes(n, cv::Size(W, H));
  std::vector<cv::Point> positions(n, cv::Point(0, 0));
  const Rig rig = make_rig(sizes, positions);

  cv::Mat labels(H, W, CV_8U);
  for (int x = 0; x < W; ++x) {
    labels.col(x).setTo(std::min(n - 1, x * n / W));
  }

  Params params;
  params.fraction = 20.0f / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, n, params);
  ASSERT_TRUE(r.error.empty()) << r.error;
  ASSERT_EQ(r.weights.channels(), n);

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < n; ++c) {
        const float w = weight_at(r.weights, x, y, c);
        EXPECT_TRUE(std::isfinite(w)) << "(" << x << "," << y << ") ch " << c;
        EXPECT_GE(w, 0.0f);
        EXPECT_LE(w, 1.0f);
      }
      EXPECT_GT(sum_at(r.weights, x, y), 0.0f) << "(" << x << "," << y << ")";
    }
  }
}

// A label with no pixels saturates one distance transform, at about 1.8e19 with the exact transform, 3.4e38 with the
// fast one, never infinity, so both forms stay finite. What this pins is that an empty region produces a usable field
// at all; the half-pixel offset that selecting preserves is pinned by TwoCameraProfileMatchesClosedForm.
TEST(FeatherMaskTest, EmptyRegionStaysFinite) {
  constexpr int W = 96, H = 16;
  const Rig rig = make_rig({{W, H}, {W, H}, {W, H}}, {{0, 0}, {0, 0}, {0, 0}});
  // Label 2 is never used.
  const cv::Mat labels = vertical_seam(W, H, 48);

  Params params;
  params.fraction = 16.0f / static_cast<float>(W);
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 3, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_TRUE(std::isfinite(weight_at(r.weights, x, y, c)));
      }
      EXPECT_FLOAT_EQ(weight_at(r.weights, x, y, 2), 0.0f) << "unused label contributed at (" << x << "," << y << ")";
      EXPECT_NEAR(sum_at(r.weights, x, y), 1.0f, 1e-6f);
    }
  }
}

// Pixels no camera maps must contribute nothing; the blend kernel emits transparent black there.
TEST(FeatherMaskTest, UncoveredPixelsGetZeroWeight) {
  constexpr int W = 200, H = 16;
  const Rig base = make_rig({{80, H}, {80, H}}, {{0, 0}, {60, 0}});
  Rig rig = base;
  // Punch an unmapped hole in camera 0 at canvas x in [20,30).
  for (int y = 0; y < H; ++y) {
    for (int x = 20; x < 30; ++x) {
      rig.remap_x[0].at<uint16_t>(y, x) = kUnmapped;
    }
  }
  cv::Mat labels(H, W, CV_8U, cv::Scalar(0));
  labels.colRange(70, W).setTo(1);

  Params params;
  params.fraction = 8.0f / 80.0f;
  const Result r = build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, params);
  ASSERT_TRUE(r.error.empty()) << r.error;

  for (int y = 0; y < H; ++y) {
    // The hole is outside camera 1's footprint too, so nothing covers it.
    for (int x = 20; x < 30; ++x) {
      EXPECT_FLOAT_EQ(sum_at(r.weights, x, y), 0.0f) << "(" << x << "," << y << ")";
    }
    // Canvas beyond camera 1's footprint is likewise uncovered.
    for (int x = 140; x < W; ++x) {
      EXPECT_FLOAT_EQ(sum_at(r.weights, x, y), 0.0f) << "(" << x << "," << y << ")";
    }
  }
}

TEST(FeatherMaskTest, RejectsMalformedInput) {
  const Rig rig = make_rig({{32, 8}, {32, 8}}, {{0, 0}, {0, 0}});
  const cv::Mat labels = vertical_seam(32, 8, 16);
  cv::Mat float_labels;
  labels.convertTo(float_labels, CV_32F);
  // Labels index the per-camera vectors directly, so an out-of-range one would read out of bounds.
  cv::Mat bad_label = labels.clone();
  bad_label.at<uint8_t>(0, 0) = 5;
  for (float fraction : {0.0f, 0.05f}) {
    Params params;
    params.fraction = fraction;
    EXPECT_FALSE(build_weights(float_labels, rig.remap_x, rig.remap_y, rig.positions, 2, params).error.empty());
    EXPECT_FALSE(build_weights(cv::Mat(), rig.remap_x, rig.remap_y, rig.positions, 2, params).error.empty());
    EXPECT_FALSE(build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 3, params).error.empty());
    EXPECT_FALSE(build_weights(bad_label, rig.remap_x, rig.remap_y, rig.positions, 2, params).error.empty());
    std::vector<cv::Mat> invalid_maps = rig.remap_x;
    invalid_maps[0].convertTo(invalid_maps[0], CV_32F);
    EXPECT_FALSE(build_weights(labels, invalid_maps, rig.remap_y, rig.positions, 2, params).error.empty());
    invalid_maps[0] = rig.remap_x[0].row(0);
    EXPECT_FALSE(build_weights(labels, invalid_maps, rig.remap_y, rig.positions, 2, params).error.empty());
  }

  Params negative;
  negative.fraction = -0.1f;
  EXPECT_FALSE(build_weights(labels, rig.remap_x, rig.remap_y, rig.positions, 2, negative).error.empty());
}

TEST(FeatherMaskTest, CoverageMasksFollowUnmappedSentinel) {
  constexpr int W = 40, H = 6;
  Rig rig = make_rig({{20, H}}, {{5, 0}});
  rig.remap_x[0].at<uint16_t>(2, 3) = kUnmapped;

  const auto coverage = coverage_masks(cv::Size(W, H), rig.remap_x, rig.remap_y, rig.positions);
  ASSERT_EQ(coverage.size(), 1u);
  EXPECT_EQ(coverage[0].at<uint8_t>(0, 0), 0); // left of the footprint
  EXPECT_EQ(coverage[0].at<uint8_t>(0, 5), 255); // first mapped column
  EXPECT_EQ(coverage[0].at<uint8_t>(2, 8), 0); // the sentinel we punched
  EXPECT_EQ(coverage[0].at<uint8_t>(0, 25), 0); // right of the footprint
}
