// Unit test for match_seam_images3 (multi-image exposure matching)

#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "cupano/pano/cudaPano3.h"

using hm::pano::match_seam_images3;

// Build a simple 2D seam indexed mask with three vertical bands: 0 | 1 | 2
static cv::Mat make_three_band_mask(int rows, int cols) {
  cv::Mat m(rows, cols, CV_8U);
  int c0 = cols / 3;
  int c1 = 2 * cols / 3;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      uchar v = (c < c0) ? 0 : ((c < c1) ? 1 : 2);
      m.at<uchar>(r, c) = v;
    }
  }
  return m;
}

// Create a solid-color BGR image of size HxW
static cv::Mat make_solid_bgr(int rows, int cols, uchar v) {
  return cv::Mat(rows, cols, CV_8UC3, cv::Scalar(v, v, v)).clone();
}

TEST(MatchSeamImages3, SolvesAdditiveOffsetsConstantImages) {
  const int H = 20, W = 30;
  // Three constant images with different intensities
  cv::Mat img0 = make_solid_bgr(H, W, 100);
  cv::Mat img1 = make_solid_bgr(H, W, 120);
  cv::Mat img2 = make_solid_bgr(H, W, 140);

  // Global seam mask with 3 vertical bands
  cv::Mat seam_indexed = make_three_band_mask(H, W);

  // Top-left placements (all at origin)
  cv::Point tl0(0, 0), tl1(0, 0), tl2(0, 0);

  // Sample N pixels near boundaries
  int N = 3;
  auto result = match_seam_images3(img0, img1, img2, seam_indexed, N, tl0, tl1, tl2, /*verbose=*/false);
  ASSERT_TRUE(result.has_value());

  // Expected per-image offsets (B,G,R): with gauge a0=0, a1=-20, a2=-40
  // Allow small tolerance since we average over strips
  auto adj = *result;
  auto expect_near_scalar = [](const cv::Scalar& s, double b, double g, double r, double eps) {
    EXPECT_NEAR(s[0], b, eps);
    EXPECT_NEAR(s[1], g, eps);
    EXPECT_NEAR(s[2], r, eps);
  };
  expect_near_scalar(adj[0], 0.0, 0.0, 0.0, 1e-3);
  expect_near_scalar(adj[1], -20.0, -20.0, -20.0, 1e-3);
  expect_near_scalar(adj[2], -40.0, -40.0, -40.0, 1e-3);
}
