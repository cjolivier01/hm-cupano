// cudaPanoN_test.cpp

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "cupano/gpu/gpu_runtime.h"
#include "cupano/pano/blendMode.h"
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

// memcmp alone reports a meaningless difference value; locate the first differing pixel instead.
// Also pins the shapes so an empty download cannot pass vacuously.
void expect_bytes_equal(const cv::Mat& a, const cv::Mat& b, int width, int height, const char* what) {
  ASSERT_EQ(a.cols, width);
  ASSERT_EQ(a.rows, height);
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(a.type(), b.type());
  const size_t bytes = a.total() * a.elemSize();
  ASSERT_EQ(bytes, b.total() * b.elemSize());
  if (std::memcmp(a.data, b.data, bytes) == 0) {
    return;
  }
  size_t first = 0;
  while (first < bytes && a.data[first] == b.data[first])
    ++first;
  const size_t pixel = first / a.elemSize();
  ADD_FAILURE() << what << ": hard and one-hot single-level output differ at byte " << first << " (pixel "
                << pixel % a.cols << "," << pixel / a.cols << "), hard=" << int(a.data[first])
                << " soft=" << int(b.data[first]);
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
    hm::pano::BlendSettings blend,
    bool minimize_blend,
    int max_output_width = 0) {
  hm::pano::cuda::CudaStitchPanoN<float4, float4> pano(
      /*batch_size=*/1,
      /*blend=*/blend,
      masks,
      /*minimize_blend=*/minimize_blend,
      /*quiet=*/true,
      /*max_output_width=*/max_output_width);
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

TEST(CudaPanoNMaxOutputWidthTest, ConstructorScalesMasksBeforeCanvasAllocation) {
  constexpr int W = 64;
  constexpr int H = 32;
  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(48, 0), cv::Point(96, 0)};
  cv::Mat seam(H, 160, CV_8U, cv::Scalar(0));
  seam.colRange(48, 96).setTo(1);
  seam.colRange(96, 160).setTo(2);
  ControlMasksN masks = make_masks(sizes, pos, seam);

  hm::pano::cuda::CudaStitchPanoN<float4, float4> pano(
      /*batch_size=*/1,
      /*num_levels=*/0,
      masks,
      /*minimize_blend=*/false,
      /*quiet=*/true,
      /*max_output_width=*/80);

  ASSERT_TRUE(pano.status().ok()) << pano.status().message();
  EXPECT_EQ(pano.canvas_width(), 80);
  EXPECT_EQ(pano.canvas_height(), 16);
}

TEST(CudaPanoNMaxOutputWidthTest, MinimizedSoftSeamUsesScaledRemapRois) {
  constexpr int W = 96;
  constexpr int H = 64;
  constexpr int CANVAS_W = 240;
  const std::vector<cv::Size> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::vector<cv::Point> pos = {cv::Point(0, 0), cv::Point(72, 0), cv::Point(144, 0)};
  cv::Mat seam(H, CANVAS_W, CV_8U, cv::Scalar(0));
  seam.colRange(72, 144).setTo(1);
  seam.colRange(144, CANVAS_W).setTo(2);
  ControlMasksN masks = make_masks(sizes, pos, seam);
  std::vector<cv::Mat> imgs = {
      make_pattern_image_f4(W, H, 0), make_pattern_image_f4(W, H, 1), make_pattern_image_f4(W, H, 2)};
  auto up = upload_inputs(imgs);

  cv::Mat capped = run_pano(masks, up.ptrs, /*num_levels=*/2, /*minimize_blend=*/true, /*max_output_width=*/120);

  ASSERT_FALSE(capped.empty());
  EXPECT_EQ(capped.cols, 120);
  EXPECT_EQ(capped.rows, 32);
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

namespace {
template <typename Pixel>
void check_compact_borrowed_4() {
  constexpr int w = 97, h = 35, n = 4;
  constexpr int canvas_width = w + 24 * (n - 1);
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  for (int x = 0; x < canvas_width; ++x)
    seam.col(x).setTo(std::min(n - 1, x * n / canvas_width));
  auto masks = make_masks({{w, h}, {w, h}, {w, h}, {w, h}}, {{0, 0}, {24, 0}, {48, 0}, {72, 0}}, seam);
  for (int levels : {1, 4}) {
    hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> reference(1, levels, masks, false, true, 0, false);
    hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> compact(1, levels, masks, false, true, 0, true);
    ASSERT_TRUE(reference.status().ok()) << reference.status().message();
    ASSERT_TRUE(compact.status().ok()) << compact.status().message();
    for (int frame = 0; frame < 3; ++frame) {
      std::vector<std::unique_ptr<hm::CudaMat<Pixel>>> inputs;
      std::vector<const hm::CudaMat<Pixel>*> ptrs;
      for (int i = 0; i < n; ++i) {
        cv::Mat host = make_pattern_image_f4(w, h, frame + i);
        if (sizeof(Pixel) == 8)
          host.convertTo(host, CV_16FC4);
        inputs.push_back(std::make_unique<hm::CudaMat<Pixel>>(host));
        ptrs.push_back(inputs.back().get());
      }
      auto expected = reference.process(ptrs, 0, nullptr);
      ASSERT_TRUE(expected.ok()) << expected.status().message();
      auto actual = compact.process(ptrs, 0, nullptr);
      ASSERT_TRUE(actual.ok()) << actual.status().message();
      CUDA_CHECK(cudaDeviceSynchronize());
      auto owned = expected.ConsumeValueOrDie();
      auto borrowed = actual.ConsumeValueOrDie();
      EXPECT_EQ(borrowed->width(), canvas_width);
      EXPECT_EQ(borrowed->height(), h);
      cv::Mat a = owned->download(), b = borrowed->download();
      // Compare raw bytes, including alpha and half precision rounding.
      ASSERT_EQ(a.total() * a.elemSize(), b.total() * b.elemSize());
      EXPECT_EQ(std::memcmp(a.data, b.data, a.total() * a.elemSize()), 0) << "frame=" << frame << " levels=" << levels;
      // Destroy the non-owning wrapper before the next frame reuses scratch.
    }
  }
}

// A one-hot seam mask blended at a single level selects exactly one contributor per pixel, so it
// must reproduce the hard-seam kernel byte for byte. The fixture below keeps every label region
// inside its owner's remap footprint, which is the precondition for that equality: where an owner
// is unmapped the hard-seam kernel leaves the memset while the blend kernel falls back to the
// highest-alpha contributor. The assertion below pins that precondition so a fixture edit fails
// with a clear message instead of a bare memcmp mismatch.
//
// This is the regression test for the soft-seam mask being uploaded with a fixed CV_32F depth while
// the blend kernels read it as BaseScalar_t<T_compute>. Half pipelines then read float bit patterns
// as pairs of __half and pick the wrong contributor across roughly half the canvas.
template <typename Pixel>
void check_single_level_matches_hard_seam_4(bool minimize_blend, int w, int h, int stride) {
  constexpr int n = 4;
  const int canvas_width = w + stride * (n - 1);
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  for (int x = 0; x < canvas_width; ++x)
    seam.col(x).setTo(std::min(n - 1, x * n / canvas_width));

  // Precondition: every labelled column lies within its owner's footprint, and the identity remaps
  // below mark no pixel unmapped, so every owner actually has data wherever it owns.
  for (int x = 0; x < canvas_width; ++x) {
    const int owner = seam.at<uint8_t>(0, x);
    ASSERT_GE(x, stride * owner) << "label " << owner << " starts before its footprint at x=" << x;
    ASSERT_LT(x, stride * owner + w) << "label " << owner << " extends past its footprint at x=" << x;
  }

  std::vector<cv::Size> sizes(n, cv::Size(w, h));
  std::vector<cv::Point> positions;
  for (int i = 0; i < n; ++i)
    positions.emplace_back(stride * i, 0);
  auto masks = make_masks(sizes, positions, seam);

  hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> hard(1, /*num_levels=*/0, masks, minimize_blend, true, 0, false);
  hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> soft(1, /*num_levels=*/1, masks, minimize_blend, true, 0, false);
  ASSERT_TRUE(hard.status().ok()) << hard.status().message();
  ASSERT_TRUE(soft.status().ok()) << soft.status().message();
  if (minimize_blend) {
    // Otherwise select_regions rejects the ROI and this degenerates into the full-canvas case,
    // leaving the cropped-seam constructor path untested.
    ASSERT_TRUE(soft.minimizes_blend()) << "fixture too small for the blend ROI to engage";
  }

  std::vector<std::unique_ptr<hm::CudaMat<Pixel>>> inputs;
  std::vector<const hm::CudaMat<Pixel>*> ptrs;
  for (int i = 0; i < n; ++i) {
    cv::Mat host = make_pattern_image_f4(w, h, i);
    if (sizeof(Pixel) == 8)
      host.convertTo(host, CV_16FC4);
    inputs.push_back(std::make_unique<hm::CudaMat<Pixel>>(host));
    ptrs.push_back(inputs.back().get());
  }

  auto hard_out =
      hard.process(ptrs, 0, std::make_unique<hm::CudaMat<Pixel>>(1, hard.canvas_width(), hard.canvas_height()));
  ASSERT_TRUE(hard_out.ok()) << hard_out.status().message();
  auto soft_out =
      soft.process(ptrs, 0, std::make_unique<hm::CudaMat<Pixel>>(1, soft.canvas_width(), soft.canvas_height()));
  ASSERT_TRUE(soft_out.ok()) << soft_out.status().message();
  CUDA_CHECK(cudaDeviceSynchronize());

  cv::Mat a = hard_out.ConsumeValueOrDie()->download();
  cv::Mat b = soft_out.ConsumeValueOrDie()->download();
  expect_bytes_equal(a, b, canvas_width, h, minimize_blend ? "minimized" : "full canvas");
}
} // namespace
TEST(CudaPanoNCompactTest, BorrowedOutputMatchesOwnedFloat4) {
  check_compact_borrowed_4<float4>();
}
TEST(CudaPanoNCompactTest, BorrowedOutputMatchesOwnedHalf4) {
  check_compact_borrowed_4<half4>();
}
// ControlMasksN derives the canvas from the remaps and positions, never from the seam mask, and
// CanvasManagerN::convertMaskMat only pads. An oversized seam therefore reaches the blend kernel,
// which indexes it with the blend buffers' stride, and silently selects the wrong contributor.
//
// Optimized builds only: convertMaskMat asserts on this first in debug builds, and that assert is
// compiled out under NDEBUG, which is precisely when the status check has to catch it.
#ifdef NDEBUG
TEST(CudaPanoNSeamMaskTest, OversizedSeamMaskIsRejected) {
  constexpr int w = 97, h = 35, n = 4, stride = 24;
  constexpr int canvas_width = w + stride * (n - 1);
  cv::Mat oversized(h + 5, canvas_width + 31, CV_8U, cv::Scalar(0));
  for (int x = 0; x < oversized.cols; ++x)
    oversized.col(x).setTo(std::min(n - 1, x * n / oversized.cols));

  std::vector<cv::Size> sizes(n, cv::Size(w, h));
  std::vector<cv::Point> positions;
  for (int i = 0; i < n; ++i)
    positions.emplace_back(stride * i, 0);
  auto masks = make_masks(sizes, positions, oversized);

  hm::pano::cuda::CudaStitchPanoN<half4, half4> soft(
      1,
      /*num_levels=*/1,
      masks,
      /*minimize_blend=*/false,
      /*quiet=*/true,
      /*max_output_width=*/0,
      /*compact_workspace=*/false);
  EXPECT_FALSE(soft.status().ok()) << "an oversized seam mask must not construct silently";
}
#endif

TEST(CudaPanoNSeamMaskTest, SingleLevelMatchesHardSeamFloat4) {
  check_single_level_matches_hard_seam_4<float4>(/*minimize_blend=*/false, /*w=*/97, /*h=*/35, /*stride=*/24);
}
TEST(CudaPanoNSeamMaskTest, SingleLevelMatchesHardSeamHalf4) {
  check_single_level_matches_hard_seam_4<half4>(/*minimize_blend=*/false, /*w=*/97, /*h=*/35, /*stride=*/24);
}
// The minimize path crops the seam before one-hotting it, so it builds the mask from a ROI view.
// This needs a canvas wide enough that select_regions does not reject the ROI for covering most of
// it; the small fixture above would silently fall back to a full-canvas blend.
TEST(CudaPanoNSeamMaskTest, SingleLevelMatchesHardSeamHalf4Minimized) {
  check_single_level_matches_hard_seam_4<half4>(/*minimize_blend=*/true, /*w=*/2000, /*h=*/200, /*stride=*/700);
}

namespace {

using hm::pano::BlendSettings;

// Overlapping strip rig: every label region sits strictly inside its owner's footprint, so the
// alpha and hard-seam paths are comparable pixel for pixel.
ControlMasksN make_strip_masks(int w, int h, int n, int stride) {
  const int canvas_width = w + stride * (n - 1);
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  for (int x = 0; x < canvas_width; ++x) {
    seam.col(x).setTo(std::min(n - 1, x * n / canvas_width));
  }
  std::vector<cv::Size> sizes(n, cv::Size(w, h));
  std::vector<cv::Point> positions;
  for (int i = 0; i < n; ++i) {
    positions.emplace_back(stride * i, 0);
  }
  return make_masks(sizes, positions, seam);
}

} // namespace

// A zero-width feather must reproduce the hard seam exactly, so turning the crossfade off is a
// no-op rather than an approximation.
TEST(CudaPanoNAlphaTest, ZeroFeatherMatchesHardSeam) {
  constexpr int w = 97, h = 35, n = 4;
  const ControlMasksN masks = make_strip_masks(w, h, n, 24);
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < n; ++i) {
    imgs.push_back(make_pattern_image_f4(w, h, i));
  }
  auto up = upload_inputs(imgs);

  const cv::Mat hard = run_pano(masks, up.ptrs, BlendSettings::HardSeam(), /*minimize_blend=*/false);
  const cv::Mat alpha = run_pano(masks, up.ptrs, BlendSettings::Alpha(0.0f), /*minimize_blend=*/false);
  ASSERT_FALSE(hard.empty());
  ASSERT_FALSE(alpha.empty());
  ASSERT_EQ(hard.total() * hard.elemSize(), alpha.total() * alpha.elemSize());
  EXPECT_EQ(std::memcmp(hard.data, alpha.data, hard.total() * hard.elemSize()), 0);
}

// A convex combination of identical inputs must return that input untouched. On float4 the N
// kernel normalizes unconditionally and drops zero-alpha contributors, so this cannot fail on the
// weights not summing to one or on an uncovered camera leaking black - both are pinned in
// featherMask_test. What it pins is that the mask reaches the kernel with the right layout.
TEST(CudaPanoNAlphaTest, ConstantInputIsPreservedExactly) {
  constexpr int w = 97, h = 35, n = 3;
  const ControlMasksN masks = make_strip_masks(w, h, n, 32);
  const cv::Vec4f colour(40.0f, 90.0f, 170.0f, 255.0f);
  std::vector<cv::Mat> imgs(n, cv::Mat(h, w, CV_32FC4, colour));
  auto up = upload_inputs(imgs);

  const cv::Mat out = run_pano(masks, up.ptrs, BlendSettings::Alpha(0.2f), /*minimize_blend=*/false);
  ASSERT_FALSE(out.empty());
  for (int y = 0; y < out.rows; ++y) {
    const cv::Vec4f* row = out.ptr<cv::Vec4f>(y);
    for (int x = 0; x < out.cols; ++x) {
      if (row[x][3] == 0.0f) {
        continue; // nothing covers this pixel
      }
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(row[x][c], colour[c], 1e-3f) << "(" << x << "," << y << ") ch " << c;
      }
    }
  }
}

// Alpha mode must actually blend: the output has to differ from the hard seam near the seam, and
// match it away from the seam where the ramp has saturated.
TEST(CudaPanoNAlphaTest, FeatherChangesOnlyTheSeamBand) {
  constexpr int w = 129, h = 24, n = 2;
  const ControlMasksN masks = make_strip_masks(w, h, n, 64);
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < n; ++i) {
    imgs.push_back(make_pattern_image_f4(w, h, i + 1));
  }
  auto up = upload_inputs(imgs);

  const cv::Mat hard = run_pano(masks, up.ptrs, BlendSettings::HardSeam(), /*minimize_blend=*/false);
  const cv::Mat alpha = run_pano(masks, up.ptrs, BlendSettings::Alpha(0.15f), /*minimize_blend=*/false);
  ASSERT_FALSE(hard.empty());
  ASSERT_FALSE(alpha.empty());

  int differing = 0;
  for (int y = 0; y < hard.rows; ++y) {
    const cv::Vec4f* a = hard.ptr<cv::Vec4f>(y);
    const cv::Vec4f* b = alpha.ptr<cv::Vec4f>(y);
    for (int x = 0; x < hard.cols; ++x) {
      if (std::abs(a[x][0] - b[x][0]) > 1e-3f) {
        ++differing;
      }
    }
  }
  EXPECT_GT(differing, 0) << "alpha mode produced no crossfade at all";
  EXPECT_LT(differing, hard.total() / 2) << "the crossfade should be confined to the seam band";
}

// The blend ROI must account for the feather band, so minimizing it cannot change the result.
TEST(CudaPanoNAlphaTest, MinimizeBlendMatchesFullCanvas) {
  // Wide enough that select_regions does not reject the ROI for covering most of the canvas;
  // a smaller fixture silently degenerates into two full-canvas runs.
  constexpr int w = 2000, h = 200, n = 3;
  const ControlMasksN masks = make_strip_masks(w, h, n, 700);
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < n; ++i) {
    imgs.push_back(make_pattern_image_f4(w, h, i));
  }
  auto up = upload_inputs(imgs);

  {
    hm::pano::cuda::CudaStitchPanoN<float4, float4> probe(1, BlendSettings::Alpha(0.1f), masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "fixture too small for the blend ROI to engage";
  }
  const cv::Mat full = run_pano(masks, up.ptrs, BlendSettings::Alpha(0.1f), /*minimize_blend=*/false);
  const cv::Mat mini = run_pano(masks, up.ptrs, BlendSettings::Alpha(0.1f), /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// The seam maximum is not an upper bound on where the band reaches. Where the cap pinches every
// seam pixel but coverage deepens away from the seam, the local radius grows with it, so the ROI
// has to be padded from the request. Canvas height matters: at h = 200 the coverage distance
// saturates near 100 px, the cap pins R below overlap_padding whatever the fraction is, and the
// pad never decides anything.
TEST(CudaPanoNAlphaTest, MinimizeBlendMatchesFullCanvasWhenTheSeamIsPinchedButTheBandIsNot) {
  constexpr int h = 900, canvas_w = 1200, seam_x = 600, seam_w = 5;
  // Camera 0 blankets the canvas; camera 1 is a narrower window whose left edge sits just left of
  // the seam, so its coverage depth at the seam is a couple of pixels and grows 1:1 to the right.
  cv::Mat seam(h, canvas_w, CV_8U, cv::Scalar(0));
  seam.colRange(seam_x, seam_x + seam_w).setTo(1);
  const ControlMasksN masks =
      make_masks({cv::Size(canvas_w, h), cv::Size(605, h)}, {cv::Point(0, 0), cv::Point(595, 0)}, seam);
  std::vector<cv::Mat> imgs{make_pattern_image_f4(canvas_w, h, 0), make_pattern_image_f4(605, h, 1)};
  auto up = upload_inputs(imgs);

  // narrowest = 605, so 0.9 asks for 544.5 and max_px clamps it to 512 against a 22 px seam.
  const BlendSettings blend = BlendSettings::Alpha(0.9f);
  {
    hm::pano::cuda::CudaStitchPanoN<float4, float4> probe(1, blend, masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "fixture too small for the blend ROI to engage";
  }
  const cv::Mat full = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/false);
  const cv::Mat mini = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// A coverage hole inside an overlap must not corrupt the minimized result. As shipped the
// corrected-label ROI grows to cover the hole, so minimizing stays on, which the probe below
// pins. This fixture does not discriminate the ROI change, though: reverting it puts the hole
// outside the ROI, the guard bails, and the comparison passes trivially. See
// MinimizeBlendCoversASeamTheCorrectionMovedInsideTheWriteRoi for the one that does.
TEST(CudaPanoNAlphaTest, MinimizeBlendSurvivesACoverageHole) {
  constexpr int w = 2000, h = 300, n = 3, stride = 1200;
  ControlMasksN masks = make_strip_masks(w, h, n, stride);
  // Camera 0 owns x < 1466 and overlaps camera 1 from x = 1200. A hole at x = 1210 is inside that
  // overlap but outside the padded seam bbox, so only the correction puts a boundary there.
  masks.img_col[0](cv::Rect(1210, 100, 110, 100)).setTo(65535);
  masks.img_row[0](cv::Rect(1210, 100, 110, 100)).setTo(65535);
  std::vector<cv::Mat> imgs;
  for (int i = 0; i < n; ++i) {
    imgs.push_back(make_pattern_image_f4(w, h, i));
  }
  auto up = upload_inputs(imgs);

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  {
    hm::pano::cuda::CudaStitchPanoN<float4, float4> probe(1, blend, masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "the corrected-label ROI should cover the hole, not bail on it";
  }
  const cv::Mat full = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/false);
  const cv::Mat mini = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// hard_baseline_covers_soft_owners_outside_write guards the hard baseline, which is built from the
// labels handed in. Checking the corrected labels instead passes vacuously, because a corrected
// owner covers by construction, and the minimized path then writes nothing where the original
// owner had no data.
TEST(CudaPanoNAlphaTest, MinimizeBlendBailsWhenTheHardBaselineHasNoDataOutsideTheWriteRoi) {
  constexpr int w = 2000, h = 200, n = 3, stride = 700;
  constexpr int canvas_w = w + stride * (n - 1);
  ControlMasksN masks = make_strip_masks(w, h, n, stride);
  // Camera 1 blankets the canvas, so every pixel is covered by something and the correction can
  // always find an owner. Camera 0 loses its left edge, which the hard baseline still labels 0.
  masks.img_col[1] = make_identity_map_x(canvas_w, h);
  masks.img_row[1] = make_identity_map_y(canvas_w, h);
  masks.positions[1] = SpatialTiff{0.0F, 0.0F};
  masks.img_col[0](cv::Rect(0, 0, 200, h)).setTo(65535);
  masks.img_row[0](cv::Rect(0, 0, 200, h)).setTo(65535);
  ASSERT_TRUE(masks.is_valid());
  std::vector<cv::Mat> imgs{
      make_pattern_image_f4(w, h, 0), make_pattern_image_f4(canvas_w, h, 1), make_pattern_image_f4(w, h, 2)};
  auto up = upload_inputs(imgs);

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  const cv::Mat full = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/false);
  const cv::Mat mini = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// The guard and the ROI are independent, and a hole that sits just inside the write ROI shows why.
// The guard only inspects pixels outside the ROI, so this hole passes it and minimize stays on.
// But correcting the hole's labels moves a seam to its edge, and the crossfade from that seam runs
// outside a ROI derived from the labels handed in. Only sizing the ROI from the corrected labels
// covers it.
TEST(CudaPanoNAlphaTest, MinimizeBlendCoversASeamTheCorrectionMovedInsideTheWriteRoi) {
  constexpr int canvas_w = 3400, h = 400, seam_x = 1700;
  // Both cameras blanket the canvas, so the corrected labels differ from the originals exactly
  // over the hole and nowhere else.
  cv::Mat seam(h, canvas_w, CV_8U, cv::Scalar(0));
  seam.colRange(seam_x, canvas_w).setTo(1);
  ControlMasksN masks =
      make_masks({cv::Size(canvas_w, h), cv::Size(canvas_w, h)}, {cv::Point(0, 0), cv::Point(0, 0)}, seam);
  // Alpha(0.05) asks for 170 px, so the pad is 128 and the original-label write ROI starts at
  // 1571. The hole starts on that exact column, which is what keeps it out of the guard's reach.
  masks.img_col[0](cv::Rect(1571, 0, 60, h)).setTo(65535);
  masks.img_row[0](cv::Rect(1571, 0, 60, h)).setTo(65535);
  std::vector<cv::Mat> imgs{make_pattern_image_f4(canvas_w, h, 0), make_pattern_image_f4(canvas_w, h, 1)};
  auto up = upload_inputs(imgs);

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  {
    hm::pano::cuda::CudaStitchPanoN<float4, float4> probe(1, blend, masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "the hole must stay inside the write ROI, not trip the guard";
  }
  const cv::Mat full = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/false);
  const cv::Mat mini = run_pano(masks, up.ptrs, blend, /*minimize_blend=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

namespace {
template <typename Pixel>
void check_alpha_compact_matches_reference() {
  constexpr int w = 97, h = 35, n = 3;
  constexpr int stride = 32;
  constexpr int canvas_width = w + stride * (n - 1);
  const ControlMasksN masks = make_strip_masks(w, h, n, stride);

  hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> reference(1, BlendSettings::Alpha(0.12f), masks, false, true, 0, false);
  hm::pano::cuda::CudaStitchPanoN<Pixel, Pixel> compact(1, BlendSettings::Alpha(0.12f), masks, false, true, 0, true);
  ASSERT_TRUE(reference.status().ok()) << reference.status().message();
  ASSERT_TRUE(compact.status().ok()) << compact.status().message();
  EXPECT_GT(reference.feather_radius_px(), 0.0f);

  for (int frame = 0; frame < 3; ++frame) {
    std::vector<std::unique_ptr<hm::CudaMat<Pixel>>> inputs;
    std::vector<const hm::CudaMat<Pixel>*> ptrs;
    for (int i = 0; i < n; ++i) {
      cv::Mat host = make_pattern_image_f4(w, h, frame + i);
      if (sizeof(Pixel) == 8)
        host.convertTo(host, CV_16FC4);
      inputs.push_back(std::make_unique<hm::CudaMat<Pixel>>(host));
      ptrs.push_back(inputs.back().get());
    }
    auto expected = reference.process(ptrs, 0, nullptr);
    ASSERT_TRUE(expected.ok()) << expected.status().message();
    auto actual = compact.process(ptrs, 0, nullptr);
    ASSERT_TRUE(actual.ok()) << actual.status().message();
    CUDA_CHECK(cudaDeviceSynchronize());
    auto owned = expected.ConsumeValueOrDie();
    auto borrowed = actual.ConsumeValueOrDie();
    EXPECT_EQ(borrowed->width(), canvas_width);
    cv::Mat a = owned->download(), b = borrowed->download();
    ASSERT_EQ(a.total() * a.elemSize(), b.total() * b.elemSize());
    EXPECT_EQ(std::memcmp(a.data, b.data, a.total() * a.elemSize()), 0) << "frame=" << frame;
  }
}
} // namespace
TEST(CudaPanoNAlphaTest, CompactMatchesReferenceFloat4) {
  check_alpha_compact_matches_reference<float4>();
}
TEST(CudaPanoNAlphaTest, CompactMatchesReferenceHalf4) {
  check_alpha_compact_matches_reference<half4>();
}

TEST(CudaPanoNAlphaTest, RejectsOutOfRangeFeather) {
  constexpr int w = 64, h = 16, n = 2;
  const ControlMasksN masks = make_strip_masks(w, h, n, 32);
  hm::pano::cuda::CudaStitchPanoN<float4, float4> pano(1, BlendSettings::Alpha(-0.5f), masks, false, true);
  EXPECT_FALSE(pano.status().ok());
}
