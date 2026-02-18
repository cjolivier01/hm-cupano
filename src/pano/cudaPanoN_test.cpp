// cudaPanoN_test.cpp

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <memory>
#include <vector>

#include "cupano/pano/controlMasksN.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPanoN.h"

using hm::CudaMat;
using hm::pano::ControlMasksN;
using hm::pano::SpatialTiff;

// ----------------------------------------------------------------------------
// Utility macro to check CUDA calls in tests and fail on error.
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                             \
  do {                                                                                               \
    cudaError_t err = (call);                                                                        \
    if (err != cudaSuccess) {                                                                        \
      FAIL() << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err); \
    }                                                                                                \
  } while (0)

namespace {

constexpr float kTol = 1e-4f;

cv::Mat make_identity_map_x(int w, int h) {
  cv::Mat mx(h, w, CV_16U);
  for (int y = 0; y < h; ++y) {
    uint16_t* row = mx.ptr<uint16_t>(y);
    for (int x = 0; x < w; ++x) {
      row[x] = static_cast<uint16_t>(x);
    }
  }
  return mx;
}

cv::Mat make_identity_map_y(int w, int h) {
  cv::Mat my(h, w, CV_16U);
  for (int y = 0; y < h; ++y) {
    uint16_t* row = my.ptr<uint16_t>(y);
    for (int x = 0; x < w; ++x) {
      row[x] = static_cast<uint16_t>(y);
    }
  }
  return my;
}

cv::Mat make_pattern_image_f4(int w, int h, int idx) {
  cv::Mat img(h, w, CV_32FC4);
  for (int y = 0; y < h; ++y) {
    cv::Vec4f* row = img.ptr<cv::Vec4f>(y);
    for (int x = 0; x < w; ++x) {
      const float base = static_cast<float>(idx * 50);
      row[x][0] = base + static_cast<float>(x % 17);
      row[x][1] = base + static_cast<float>(y % 19);
      row[x][2] = base + static_cast<float>((x + y) % 23);
      row[x][3] = 255.0f;
    }
  }
  return img;
}

float max_abs_diff(const cv::Mat& a, const cv::Mat& b, cv::Point* loc_out, int* ch_out) {
  if (loc_out) {
    *loc_out = {};
  }
  if (ch_out) {
    *ch_out = 0;
  }

  CV_Assert(a.size() == b.size());
  CV_Assert(a.type() == b.type());
  CV_Assert(a.type() == CV_32FC4);

  float max_d = 0.0f;
  for (int y = 0; y < a.rows; ++y) {
    const cv::Vec4f* ra = a.ptr<cv::Vec4f>(y);
    const cv::Vec4f* rb = b.ptr<cv::Vec4f>(y);
    for (int x = 0; x < a.cols; ++x) {
      for (int c = 0; c < 4; ++c) {
        const float d = std::abs(ra[x][c] - rb[x][c]);
        if (d > max_d) {
          max_d = d;
          if (loc_out) {
            *loc_out = cv::Point(x, y);
          }
          if (ch_out) {
            *ch_out = c;
          }
        }
      }
    }
  }
  return max_d;
}

void expect_mats_near(const cv::Mat& a, const cv::Mat& b, float tol) {
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(a.type(), b.type());
  cv::Point loc;
  int ch = 0;
  const float d = max_abs_diff(a, b, &loc, &ch);
  EXPECT_LE(d, tol) << "Max abs diff " << d << " at (" << loc.x << "," << loc.y << ") ch " << ch;
}

ControlMasksN make_masks(
    const std::vector<cv::Size>& sizes,
    const std::vector<cv::Point>& positions,
    const cv::Mat& seam_index) {
  const int n = static_cast<int>(sizes.size());
  ControlMasksN m;
  m.img_col.resize(n);
  m.img_row.resize(n);
  m.positions.resize(n);
  for (int i = 0; i < n; ++i) {
    m.img_col[i] = make_identity_map_x(sizes[i].width, sizes[i].height);
    m.img_row[i] = make_identity_map_y(sizes[i].width, sizes[i].height);
    m.positions[i] = SpatialTiff{static_cast<float>(positions[i].x), static_cast<float>(positions[i].y)};
  }
  m.whole_seam_mask_indexed = seam_index.clone();
  EXPECT_TRUE(m.is_valid());
  return m;
}

struct UploadedInputs {
  std::vector<std::unique_ptr<CudaMat<float4>>> mats;
  std::vector<const CudaMat<float4>*> ptrs;
};

UploadedInputs upload_inputs(const std::vector<cv::Mat>& imgs) {
  UploadedInputs up;
  up.mats.reserve(imgs.size());
  up.ptrs.reserve(imgs.size());
  for (const auto& img : imgs) {
    up.mats.emplace_back(std::make_unique<CudaMat<float4>>(img));
    up.ptrs.push_back(up.mats.back().get());
  }
  return up;
}

cv::Mat run_pano(
    const ControlMasksN& masks,
    const std::vector<const CudaMat<float4>*>& inputs,
    int num_levels,
    bool minimize_blend) {
  hm::pano::cuda::CudaStitchPanoN<float4, float4> pano(
      /*batch_size=*/1,
      /*num_levels=*/num_levels,
      masks,
      /*match_exposure=*/false,
      /*quiet=*/true,
      /*minimize_blend=*/minimize_blend);
  if (!pano.status().ok()) {
    ADD_FAILURE() << pano.status().message();
    return {};
  }

  auto canvas = std::make_unique<CudaMat<float4>>(/*batch_size=*/1, pano.canvas_width(), pano.canvas_height());
  auto out_or = pano.process(inputs, /*stream=*/0, std::move(canvas));
  if (!out_or.ok()) {
    ADD_FAILURE() << out_or.status().message();
    return {};
  }
  auto out = out_or.ConsumeValueOrDie();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    ADD_FAILURE() << "CUDA error: " << cudaGetErrorString(err);
    return {};
  }
  return out->download();
}

void expect_rect_alpha_zero(const cv::Mat& img, const cv::Rect& r) {
  ASSERT_EQ(img.type(), CV_32FC4);
  for (int y = r.y; y < r.y + r.height; ++y) {
    const cv::Vec4f* row = img.ptr<cv::Vec4f>(y);
    for (int x = r.x; x < r.x + r.width; ++x) {
      EXPECT_NEAR(row[x][3], 0.0f, kTol) << "Non-zero alpha at (" << x << "," << y << ")";
    }
  }
}

} // namespace

TEST(CudaPanoNMinimizeBlendTest, TwoWayOverlapSeamMatchesFullBlend) {
  constexpr int W = 384;
  constexpr int H = 384;
  constexpr int L = 4;

  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(0, 0)};

  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 2, W).setTo(1);

  ControlMasksN masks = make_masks(sizes, pos, seam);
  std::vector<cv::Mat> imgs = {make_pattern_image_f4(W, H, 0), make_pattern_image_f4(W, H, 1)};
  auto up = upload_inputs(imgs);

  cv::Mat full = run_pano(masks, up.ptrs, L, /*minimize_blend=*/false);
  cv::Mat mini = run_pano(masks, up.ptrs, L, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

TEST(CudaPanoNMinimizeBlendTest, NoOverlapSeamMatchesFullBlend) {
  constexpr int W = 192;
  constexpr int H = 384;
  constexpr int L = 4;

  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(W, 0)}; // butt with no overlap

  cv::Mat seam(H, W * 2, CV_8U, cv::Scalar(0));
  seam.colRange(W, W * 2).setTo(1);

  ControlMasksN masks = make_masks(sizes, pos, seam);
  std::vector<cv::Mat> imgs = {make_pattern_image_f4(W, H, 0), make_pattern_image_f4(W, H, 1)};
  auto up = upload_inputs(imgs);

  cv::Mat full = run_pano(masks, up.ptrs, L, /*minimize_blend=*/false);
  cv::Mat mini = run_pano(masks, up.ptrs, L, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

TEST(CudaPanoNMinimizeBlendTest, GapProducesZerosAndMatchesFullBlend) {
  constexpr int W = 128;
  constexpr int H = 384;
  constexpr int L = 4;
  constexpr int CANVAS_W = 384;
  static_assert(CANVAS_W == 3 * W);

  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(2 * W, 0)}; // gap in the middle [W..2W)

  cv::Mat seam(H, CANVAS_W, CV_8U, cv::Scalar(0));
  seam.colRange(CANVAS_W / 2, CANVAS_W).setTo(1);

  ControlMasksN masks = make_masks(sizes, pos, seam);
  std::vector<cv::Mat> imgs = {make_pattern_image_f4(W, H, 0), make_pattern_image_f4(W, H, 1)};
  auto up = upload_inputs(imgs);

  cv::Mat full = run_pano(masks, up.ptrs, L, /*minimize_blend=*/false);
  cv::Mat mini = run_pano(masks, up.ptrs, L, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);

  // The uncovered gap [W..2W) must stay fully transparent (alpha==0).
  expect_rect_alpha_zero(mini, cv::Rect(W, 0, W, H));
}

TEST(CudaPanoNMinimizeBlendTest, MultiSeamIntersectionMatchesFullBlend) {
  constexpr int W = 384;
  constexpr int H = 384;
  constexpr int L = 4;
  constexpr int N = 4;

  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};

  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  const int cx = W / 2;
  const int cy = H / 2;
  const int half = 32;
  const cv::Rect region(cx - half, cy - half, 2 * half, 2 * half);
  for (int y = region.y; y < region.y + region.height; ++y) {
    uint8_t* row = seam.ptr<uint8_t>(y);
    for (int x = region.x; x < region.x + region.width; ++x) {
      const int qx = (x < cx) ? 0 : 1;
      const int qy = (y < cy) ? 0 : 1;
      row[x] = static_cast<uint8_t>(qy * 2 + qx); // 0..3
    }
  }

  ControlMasksN masks = make_masks(sizes, pos, seam);
  std::vector<cv::Mat> imgs;
  imgs.reserve(N);
  for (int i = 0; i < N; ++i) {
    imgs.push_back(make_pattern_image_f4(W, H, i));
  }
  auto up = upload_inputs(imgs);

  cv::Mat full = run_pano(masks, up.ptrs, L, /*minimize_blend=*/false);
  cv::Mat mini = run_pano(masks, up.ptrs, L, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}
