#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "cupano/gpu/gpu_runtime.h"
#include "cupano/pano/controlMasks.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano.h"

using hm::CudaMat;
using hm::pano::ControlMasks;
using hm::pano::SpatialTiff;

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
    auto* row = mx.ptr<uint16_t>(y);
    for (int x = 0; x < w; ++x) {
      row[x] = static_cast<uint16_t>(x);
    }
  }
  return mx;
}

cv::Mat make_identity_map_y(int w, int h) {
  cv::Mat my(h, w, CV_16U);
  for (int y = 0; y < h; ++y) {
    auto* row = my.ptr<uint16_t>(y);
    for (int x = 0; x < w; ++x) {
      row[x] = static_cast<uint16_t>(y);
    }
  }
  return my;
}

cv::Mat make_pattern_image_f4(int w, int h, int idx) {
  cv::Mat img(h, w, CV_32FC4);
  for (int y = 0; y < h; ++y) {
    auto* row = img.ptr<cv::Vec4f>(y);
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

float max_abs_diff(const cv::Mat& a, const cv::Mat& b) {
  CV_Assert(a.size() == b.size());
  CV_Assert(a.type() == b.type());
  CV_Assert(a.type() == CV_32FC4);

  float max_d = 0.0f;
  for (int y = 0; y < a.rows; ++y) {
    const auto* ra = a.ptr<cv::Vec4f>(y);
    const auto* rb = b.ptr<cv::Vec4f>(y);
    for (int x = 0; x < a.cols; ++x) {
      for (int c = 0; c < 4; ++c) {
        max_d = std::max(max_d, std::abs(ra[x][c] - rb[x][c]));
      }
    }
  }
  return max_d;
}

void expect_mats_near(const cv::Mat& a, const cv::Mat& b, float tol) {
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(a.type(), b.type());
  EXPECT_LE(max_abs_diff(a, b), tol);
}

ControlMasks make_masks(int width, int height, int x2, const cv::Mat& seam) {
  ControlMasks masks;
  masks.img1_col = make_identity_map_x(width, height);
  masks.img1_row = make_identity_map_y(width, height);
  masks.img2_col = make_identity_map_x(width, height);
  masks.img2_row = make_identity_map_y(width, height);
  masks.whole_seam_mask_image = seam.clone();
  masks.positions = {
      SpatialTiff{0.0f, 0.0f},
      SpatialTiff{static_cast<float>(x2), 0.0f},
  };
  EXPECT_TRUE(masks.is_valid());
  return masks;
}

struct LevelSize {
  int width{0};
  int height{0};
};

LevelSize read_level_0_size(const std::filesystem::path& metadata_path) {
  std::ifstream input(metadata_path);
  EXPECT_TRUE(input.good()) << "Unable to open " << metadata_path.string();
  std::string line;
  while (std::getline(input, line)) {
    if (line.rfind("level_0=", 0) == 0) {
      const auto dims = line.substr(std::string("level_0=").size());
      const auto pos = dims.find('x');
      EXPECT_NE(pos, std::string::npos);
      return LevelSize{
          .width = std::stoi(dims.substr(0, pos)),
          .height = std::stoi(dims.substr(pos + 1)),
      };
    }
  }
  ADD_FAILURE() << "Missing level_0 entry in " << metadata_path.string();
  return {};
}

std::filesystem::path make_temp_dir(const std::string& label) {
  const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path dir =
      std::filesystem::temp_directory_path() / (label + "_" + std::to_string(static_cast<long long>(stamp)));
  std::filesystem::create_directories(dir);
  return dir;
}

} // namespace

TEST(CudaPanoMaxOutputWidthTest, ConstructorScalesMasksBeforeCanvasAllocation) {
  constexpr int kWidth = 96;
  constexpr int kHeight = 32;
  constexpr int kX2 = 64;
  constexpr int kCanvasWidth = kWidth + kX2;
  constexpr int kMaxOutputWidth = kCanvasWidth / 2;

  cv::Mat seam(kHeight, kCanvasWidth, CV_8U, cv::Scalar(0));
  seam.colRange(0, kCanvasWidth / 2).setTo(1);
  ControlMasks masks = make_masks(kWidth, kHeight, kX2, seam);

  hm::pano::cuda::CudaStitchPano<float4, float4> pano(
      /*batch_size=*/1,
      /*num_levels=*/0,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/true,
      /*max_output_width=*/kMaxOutputWidth);

  ASSERT_TRUE(pano.status().ok()) << pano.status().message();
  EXPECT_EQ(pano.canvas_width(), kMaxOutputWidth);
  EXPECT_EQ(pano.canvas_height(), kHeight / 2);
  EXPECT_EQ(masks.canvas_width(), static_cast<size_t>(kCanvasWidth));
  EXPECT_EQ(masks.canvas_height(), static_cast<size_t>(kHeight));
}

TEST(CudaPanoMinimizeBlendTest, TwoImageFlagChangesWorkspaceSizeAndPreservesOutput) {
  constexpr int kWidth = 384;
  constexpr int kHeight = 64;
  constexpr int kX2 = 192;
  constexpr int kLevels = 4;
  const int canvas_width = kWidth + kX2;

  cv::Mat seam(kHeight, canvas_width, CV_8U, cv::Scalar(0));
  seam.colRange(0, canvas_width / 2).setTo(1);
  ControlMasks masks = make_masks(kWidth, kHeight, kX2, seam);

  CudaMat<float4> input_left(make_pattern_image_f4(kWidth, kHeight, 0));
  CudaMat<float4> input_right(make_pattern_image_f4(kWidth, kHeight, 1));

  hm::pano::cuda::CudaStitchPano<float4, float4> pano_full(
      /*batch_size=*/1,
      /*num_levels=*/kLevels,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/false);
  hm::pano::cuda::CudaStitchPano<float4, float4> pano_mini(
      /*batch_size=*/1,
      /*num_levels=*/kLevels,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/true);

  ASSERT_TRUE(pano_full.status().ok()) << pano_full.status().message();
  ASSERT_TRUE(pano_mini.status().ok()) << pano_mini.status().message();

  auto canvas_full = std::make_unique<CudaMat<float4>>(1, pano_full.canvas_width(), pano_full.canvas_height());
  auto canvas_mini = std::make_unique<CudaMat<float4>>(1, pano_mini.canvas_width(), pano_mini.canvas_height());

  auto out_full_or = pano_full.process(input_left, input_right, /*stream=*/0, std::move(canvas_full));
  ASSERT_TRUE(out_full_or.ok()) << out_full_or.status().message();
  auto out_mini_or = pano_mini.process(input_left, input_right, /*stream=*/0, std::move(canvas_mini));
  ASSERT_TRUE(out_mini_or.ok()) << out_mini_or.status().message();

  CUDA_CHECK(cudaDeviceSynchronize());

  const cv::Mat out_full = out_full_or.ConsumeValueOrDie()->download();
  const cv::Mat out_mini = out_mini_or.ConsumeValueOrDie()->download();
  expect_mats_near(out_full, out_mini, kTol);

  const auto full_dir = make_temp_dir("cuda_pano_full");
  const auto mini_dir = make_temp_dir("cuda_pano_mini");
  const auto cleanup = [&]() {
    std::error_code ec;
    std::filesystem::remove_all(full_dir, ec);
    std::filesystem::remove_all(mini_dir, ec);
  };

  ASSERT_TRUE(pano_full.dump_soft_blend_pyramid(full_dir.string(), /*stream=*/0).ok());
  ASSERT_TRUE(pano_mini.dump_soft_blend_pyramid(mini_dir.string(), /*stream=*/0).ok());

  const LevelSize full_size = read_level_0_size(full_dir / "metadata.txt");
  const LevelSize mini_size = read_level_0_size(mini_dir / "metadata.txt");
  EXPECT_EQ(full_size.width, canvas_width);
  EXPECT_EQ(full_size.height, kHeight);
  EXPECT_LT(mini_size.width, full_size.width);
  EXPECT_EQ(mini_size.height, kHeight);

  cleanup();
}

TEST(CudaPanoMinimizeBlendTest, CroppedUcharComputeSeamIsContiguous) {
  constexpr int kWidth = 384;
  constexpr int kHeight = 64;
  constexpr int kX2 = 192;
  constexpr int kLevels = 4;
  const int canvas_width = kWidth + kX2;

  cv::Mat seam(kHeight, canvas_width, CV_8U, cv::Scalar(0));
  seam.colRange(0, canvas_width / 2).setTo(1);
  ControlMasks masks = make_masks(kWidth, kHeight, kX2, seam);

  hm::pano::cuda::CudaStitchPano<uchar3, uchar3> pano(
      /*batch_size=*/1,
      /*num_levels=*/kLevels,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/true);

  ASSERT_TRUE(pano.status().ok()) << pano.status().message();
  EXPECT_TRUE(pano.minimizes_blend());
}

TEST(CudaPanoMinimizeBlendTest, NoOverlapFallsBackToFullCanvasBlend) {
  constexpr int kWidth = 64;
  constexpr int kHeight = 32;
  constexpr int kCanvasWidth = kWidth * 2;
  constexpr int kLevels = 2;

  for (const bool reversed_order : {false, true}) {
    SCOPED_TRACE(reversed_order ? "reversed disjoint ordering" : "left-to-right disjoint ordering");
    const int x1 = reversed_order ? kWidth : 0;
    const int x2 = reversed_order ? 0 : kWidth;

    cv::Mat seam(kHeight, kCanvasWidth, CV_8U, cv::Scalar(0));
    seam.colRange(x1, x1 + kWidth).setTo(1);
    ControlMasks masks = make_masks(kWidth, kHeight, x2, seam);
    masks.positions[0].xpos = static_cast<float>(x1);

    const cv::Mat host_image_1 = make_pattern_image_f4(kWidth, kHeight, 0);
    const cv::Mat host_image_2 = make_pattern_image_f4(kWidth, kHeight, 1);
    CudaMat<float4> input_image_1(host_image_1);
    CudaMat<float4> input_image_2(host_image_2);

    hm::pano::cuda::CudaStitchPano<float4, float4> pano_hard(
        /*batch_size=*/1,
        /*num_levels=*/0,
        masks,
        /*quiet=*/true,
        /*minimize_blend=*/true);
    hm::pano::cuda::CudaStitchPano<float4, float4> pano_full(
        /*batch_size=*/1,
        /*num_levels=*/kLevels,
        masks,
        /*quiet=*/true,
        /*minimize_blend=*/false);
    hm::pano::cuda::CudaStitchPano<float4, float4> pano_requested_mini(
        /*batch_size=*/1,
        /*num_levels=*/kLevels,
        masks,
        /*quiet=*/true,
        /*minimize_blend=*/true);

    ASSERT_TRUE(pano_hard.status().ok()) << pano_hard.status().message();
    ASSERT_TRUE(pano_full.status().ok()) << pano_full.status().message();
    ASSERT_TRUE(pano_requested_mini.status().ok()) << pano_requested_mini.status().message();

    auto hard_canvas = std::make_unique<CudaMat<float4>>(1, pano_hard.canvas_width(), pano_hard.canvas_height());
    auto full_canvas = std::make_unique<CudaMat<float4>>(1, pano_full.canvas_width(), pano_full.canvas_height());
    auto requested_mini_canvas =
        std::make_unique<CudaMat<float4>>(1, pano_requested_mini.canvas_width(), pano_requested_mini.canvas_height());

    auto hard_out_or = pano_hard.process(input_image_1, input_image_2, /*stream=*/0, std::move(hard_canvas));
    ASSERT_TRUE(hard_out_or.ok()) << hard_out_or.status().message();
    auto full_out_or = pano_full.process(input_image_1, input_image_2, /*stream=*/0, std::move(full_canvas));
    ASSERT_TRUE(full_out_or.ok()) << full_out_or.status().message();
    auto requested_mini_out_or =
        pano_requested_mini.process(input_image_1, input_image_2, /*stream=*/0, std::move(requested_mini_canvas));
    ASSERT_TRUE(requested_mini_out_or.ok()) << requested_mini_out_or.status().message();

    CUDA_CHECK(cudaDeviceSynchronize());

    const cv::Mat hard_out = hard_out_or.ConsumeValueOrDie()->download();
    const cv::Mat full_out = full_out_or.ConsumeValueOrDie()->download();
    const cv::Mat requested_mini_out = requested_mini_out_or.ConsumeValueOrDie()->download();
    cv::Mat expected_hard(kHeight, kCanvasWidth, CV_32FC4, cv::Scalar(0, 0, 0, 0));
    host_image_1.copyTo(expected_hard.colRange(x1, x1 + kWidth));
    host_image_2.copyTo(expected_hard.colRange(x2, x2 + kWidth));
    expect_mats_near(hard_out, expected_hard, kTol);
    expect_mats_near(requested_mini_out, full_out, kTol);

    const auto fallback_dir =
        make_temp_dir(reversed_order ? "cuda_pano_no_overlap_fallback_reversed" : "cuda_pano_no_overlap_fallback");
    ASSERT_TRUE(pano_requested_mini.dump_soft_blend_pyramid(fallback_dir.string(), /*stream=*/0).ok());
    const LevelSize fallback_size = read_level_0_size(fallback_dir / "metadata.txt");
    EXPECT_EQ(fallback_size.width, kCanvasWidth);
    EXPECT_EQ(fallback_size.height, kHeight);
    std::error_code ec;
    std::filesystem::remove_all(fallback_dir, ec);
  }
}

namespace {
template <typename Pixel>
void check_compact_borrowed_2() {
  constexpr int w = 97, h = 35, n = 2;
  constexpr int canvas_width = w + 24 * (n - 1);
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  for (int x = 0; x < canvas_width; ++x)
    seam.col(x).setTo(x < canvas_width / 2 ? 1 : 0);
  auto masks = make_masks(w, h, 24, seam);
  for (int levels : {1, 4}) {
    hm::pano::cuda::CudaStitchPano<Pixel, Pixel> reference(1, levels, masks, true, false, 0, false);
    hm::pano::cuda::CudaStitchPano<Pixel, Pixel> compact(1, levels, masks, true, false, 0, true);
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
      auto expected = reference.process(*inputs[0], *inputs[1], 0, nullptr);
      ASSERT_TRUE(expected.ok()) << expected.status().message();
      auto actual = compact.process(*inputs[0], *inputs[1], 0, nullptr);
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

// Lock-step guard for the soft-seam mask scalar depth (see cudaPanoN's equivalent). A binary mask
// blended at a single level picks exactly one contributor per pixel, so it must reproduce the
// hard-seam kernel byte for byte. This fails if the mask is ever uploaded at a depth the blend
// kernel does not read it at.
template <typename Pixel>
void check_single_level_matches_hard_seam_2() {
  constexpr int w = 97, h = 35, n = 2, stride = 24;
  constexpr int canvas_width = w + stride * (n - 1);
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  // Label 1 owns image 1 (footprint [0, w)), label 0 owns image 2 (footprint [stride, canvas)).
  for (int x = 0; x < canvas_width; ++x)
    seam.col(x).setTo(x < canvas_width / 2 ? 1 : 0);
  for (int x = 0; x < canvas_width; ++x) {
    if (seam.at<uint8_t>(0, x) == 1) {
      ASSERT_LT(x, w) << "image 1 label extends past its footprint at x=" << x;
    } else {
      ASSERT_GE(x, stride) << "image 2 label starts before its footprint at x=" << x;
    }
  }
  auto masks = make_masks(w, h, stride, seam);

  hm::pano::cuda::CudaStitchPano<Pixel, Pixel> hard(
      1,
      /*num_levels=*/0,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/false,
      /*max_output_width=*/0,
      /*compact_workspace=*/false);
  hm::pano::cuda::CudaStitchPano<Pixel, Pixel> soft(
      1,
      /*num_levels=*/1,
      masks,
      /*quiet=*/true,
      /*minimize_blend=*/false,
      /*max_output_width=*/0,
      /*compact_workspace=*/false);
  ASSERT_TRUE(hard.status().ok()) << hard.status().message();
  ASSERT_TRUE(soft.status().ok()) << soft.status().message();

  std::vector<std::unique_ptr<hm::CudaMat<Pixel>>> inputs;
  for (int i = 0; i < n; ++i) {
    cv::Mat host = make_pattern_image_f4(w, h, i);
    if (sizeof(Pixel) == 8)
      host.convertTo(host, CV_16FC4);
    inputs.push_back(std::make_unique<hm::CudaMat<Pixel>>(host));
  }

  auto hard_out = hard.process(
      *inputs[0], *inputs[1], 0, std::make_unique<hm::CudaMat<Pixel>>(1, hard.canvas_width(), hard.canvas_height()));
  ASSERT_TRUE(hard_out.ok()) << hard_out.status().message();
  auto soft_out = soft.process(
      *inputs[0], *inputs[1], 0, std::make_unique<hm::CudaMat<Pixel>>(1, soft.canvas_width(), soft.canvas_height()));
  ASSERT_TRUE(soft_out.ok()) << soft_out.status().message();
  CUDA_CHECK(cudaDeviceSynchronize());

  cv::Mat a = hard_out.ConsumeValueOrDie()->download();
  cv::Mat b = soft_out.ConsumeValueOrDie()->download();
  ASSERT_EQ(a.cols, canvas_width);
  ASSERT_EQ(a.rows, h);
  ASSERT_EQ(a.size(), b.size());
  ASSERT_EQ(a.type(), b.type());
  const size_t bytes = a.total() * a.elemSize();
  ASSERT_EQ(bytes, b.total() * b.elemSize());
  if (std::memcmp(a.data, b.data, bytes) != 0) {
    size_t first = 0;
    while (first < bytes && a.data[first] == b.data[first])
      ++first;
    const size_t pixel = first / a.elemSize();
    ADD_FAILURE() << "hard and binary single-level output differ at byte " << first << " (pixel " << pixel % a.cols
                  << "," << pixel / a.cols << "), hard=" << int(a.data[first]) << " soft=" << int(b.data[first]);
  }
}
} // namespace
TEST(CudaPanoSeamMaskTest, SingleLevelMatchesHardSeamFloat4) {
  check_single_level_matches_hard_seam_2<float4>();
}
TEST(CudaPanoSeamMaskTest, SingleLevelMatchesHardSeamHalf4) {
  check_single_level_matches_hard_seam_2<half4>();
}
TEST(CudaPanoCompactTest, BorrowedOutputMatchesOwnedFloat4) {
  check_compact_borrowed_2<float4>();
}
TEST(CudaPanoCompactTest, BorrowedOutputMatchesOwnedHalf4) {
  check_compact_borrowed_2<half4>();
}

TEST(CudaPanoCompactTest, InvalidContextDumpReturnsError) {
  hm::pano::cuda::CudaStitchPano<half4, half4> pano(1, 4, ControlMasks{}, true, false, 0, true);
  EXPECT_FALSE(pano.status().ok());
  EXPECT_FALSE(pano.dump_soft_blend_pyramid("/unused-invalid-context", nullptr).ok());
}

namespace {

using hm::pano::BlendSettings;

// Two overlapping cameras with the seam inside the overlap, so the crossfade has data on both
// sides. Label 1 owns image 1, label 0 owns image 2.
ControlMasks make_overlap_masks(int w, int h, int stride, int seam_x) {
  const int canvas_width = w + stride;
  cv::Mat seam(h, canvas_width, CV_8U, cv::Scalar(0));
  seam.colRange(0, seam_x).setTo(1);
  return make_masks(w, h, stride, seam);
}

template <typename Pixel>
std::unique_ptr<hm::CudaMat<Pixel>> upload(const cv::Mat& host_f4) {
  cv::Mat host = host_f4.clone();
  if (sizeof(Pixel) == 8)
    host.convertTo(host, CV_16FC4);
  return std::make_unique<hm::CudaMat<Pixel>>(host);
}

cv::Mat run_two(const ControlMasks& masks, BlendSettings blend, bool minimize_blend, const std::vector<cv::Mat>& imgs) {
  hm::pano::cuda::CudaStitchPano<float4, float4> pano(1, blend, masks, true, minimize_blend, 0, false);
  EXPECT_TRUE(pano.status().ok()) << pano.status().message();
  if (!pano.status().ok())
    return {};
  auto a = upload<float4>(imgs[0]);
  auto b = upload<float4>(imgs[1]);
  auto out =
      pano.process(*a, *b, 0, std::make_unique<hm::CudaMat<float4>>(1, pano.canvas_width(), pano.canvas_height()));
  EXPECT_TRUE(out.ok()) << out.status().message();
  if (!out.ok())
    return {};
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  return out.ConsumeValueOrDie()->download();
}

} // namespace

TEST(CudaPanoAlphaTest, RejectsIntegralCompute) {
  const ControlMasks masks = make_overlap_masks(97, 35, 24, 60);
  hm::pano::cuda::CudaStitchPano<uchar3, uchar3> pano(1, BlendSettings::Alpha(0.2f), masks, true, false);
  EXPECT_EQ(pano.status().code(), cudaErrorNotSupported);
}

// A zero-width feather must reproduce the hard seam exactly.
TEST(CudaPanoAlphaTest, ZeroFeatherMatchesHardSeam) {
  constexpr int w = 97, h = 35, stride = 24;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 60);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 0), make_pattern_image_f4(w, h, 1)};

  const cv::Mat hard = run_two(masks, BlendSettings::HardSeam(), false, imgs);
  const cv::Mat alpha = run_two(masks, BlendSettings::Alpha(0.0f), false, imgs);
  ASSERT_FALSE(hard.empty());
  ASSERT_FALSE(alpha.empty());
  ASSERT_EQ(hard.total() * hard.elemSize(), alpha.total() * alpha.elemSize());
  EXPECT_EQ(std::memcmp(hard.data, alpha.data, hard.total() * hard.elemSize()), 0);
}

// A convex combination of identical inputs must return that input untouched.
TEST(CudaPanoAlphaTest, ConstantInputIsPreservedExactly) {
  constexpr int w = 97, h = 35, stride = 24;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 60);
  const cv::Vec4f colour(30.0f, 120.0f, 200.0f, 255.0f);
  const std::vector<cv::Mat> imgs = {cv::Mat(h, w, CV_32FC4, colour), cv::Mat(h, w, CV_32FC4, colour)};

  const cv::Mat out = run_two(masks, BlendSettings::Alpha(0.2f), false, imgs);
  ASSERT_FALSE(out.empty());
  for (int y = 0; y < out.rows; ++y) {
    const cv::Vec4f* row = out.ptr<cv::Vec4f>(y);
    for (int x = 0; x < out.cols; ++x) {
      if (row[x][3] == 0.0f)
        continue;
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(row[x][c], colour[c], 1e-3f) << "(" << x << "," << y << ") ch " << c;
      }
    }
  }
}

// The crossfade must exist, and must stay near the seam.
TEST(CudaPanoAlphaTest, FeatherChangesOnlyTheSeamBand) {
  constexpr int w = 97, h = 35, stride = 24;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 60);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 1), make_pattern_image_f4(w, h, 2)};

  const cv::Mat hard = run_two(masks, BlendSettings::HardSeam(), false, imgs);
  const cv::Mat alpha = run_two(masks, BlendSettings::Alpha(0.15f), false, imgs);
  ASSERT_FALSE(hard.empty());
  ASSERT_FALSE(alpha.empty());

  size_t differing = 0;
  for (int y = 0; y < hard.rows; ++y) {
    const cv::Vec4f* a = hard.ptr<cv::Vec4f>(y);
    const cv::Vec4f* b = alpha.ptr<cv::Vec4f>(y);
    for (int x = 0; x < hard.cols; ++x) {
      if (std::abs(a[x][0] - b[x][0]) > 1e-3f)
        ++differing;
    }
  }
  EXPECT_GT(differing, 0u) << "alpha mode produced no crossfade at all";
  EXPECT_LT(differing, hard.total() / 2) << "the crossfade should be confined to the seam band";
}

TEST(CudaPanoAlphaTest, MinimizeBlendMatchesFullCanvas) {
  // Wide enough that select_regions does not reject the ROI for covering most of the canvas.
  constexpr int w = 2000, h = 200, stride = 900;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 1400);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 0), make_pattern_image_f4(w, h, 1)};

  {
    hm::pano::cuda::CudaStitchPano<float4, float4> probe(1, BlendSettings::Alpha(0.1f), masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "fixture too small for the blend ROI to engage";
  }
  const cv::Mat full = run_two(masks, BlendSettings::Alpha(0.1f), false, imgs);
  const cv::Mat mini = run_two(masks, BlendSettings::Alpha(0.1f), true, imgs);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// The guard and the ROI padding are written out per stitcher rather than shared, so the N-camera
// coverage does not protect this path.

// Correcting a coverage hole moves a seam to its edge. The guard only inspects pixels outside the
// write ROI, so a hole that lands inside it passes, and the crossfade from the moved seam then
// runs outside a ROI derived from the labels handed in.
TEST(CudaPanoAlphaTest, MinimizeBlendCoversASeamTheCorrectionMovedInsideTheWriteRoi) {
  constexpr int w = 3400, h = 400, seam_x = 1700;
  // Both cameras blanket, so the corrected labels differ from the originals exactly over the hole.
  cv::Mat seam(h, w, CV_8U, cv::Scalar(0));
  seam.colRange(0, seam_x).setTo(1);
  ControlMasks masks = make_masks(w, h, 0, seam);
  // Alpha(0.05) asks for 170 px, so the pad is 128 and the write ROI starts at 1572. Label 1 owns
  // x < 1700, so the hole goes in image 1, starting on the ROI's first column.
  masks.img1_col(cv::Rect(1572, 0, 60, h)).setTo(65535);
  masks.img1_row(cv::Rect(1572, 0, 60, h)).setTo(65535);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 0), make_pattern_image_f4(w, h, 1)};

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  {
    hm::pano::cuda::CudaStitchPano<float4, float4> probe(1, blend, masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "the hole must stay inside the write ROI, not trip the guard";
  }
  const cv::Mat full = run_two(masks, blend, false, imgs);
  const cv::Mat mini = run_two(masks, blend, true, imgs);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// Outside the write ROI the minimized path keeps the hard baseline, which is remapped from the
// labels handed in. Where the labelled camera has no data there, minimizing has to be abandoned.
TEST(CudaPanoAlphaTest, MinimizeBlendBailsWhenTheHardBaselineHasNoDataOutsideTheWriteRoi) {
  constexpr int w = 2000, h = 200, stride = 900, seam_x = 1700;
  ControlMasks masks = make_overlap_masks(w, h, stride, seam_x);
  // Label 0 owns x >= 1700 and is image 2, which starts at 900. The hole begins at the seam so
  // the corrected seam moves to its far edge and most of the hole stays outside the ROI.
  masks.img2_col(cv::Rect(seam_x - stride, 0, 700, h)).setTo(65535);
  masks.img2_row(cv::Rect(seam_x - stride, 0, 700, h)).setTo(65535);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 0), make_pattern_image_f4(w, h, 1)};

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  const cv::Mat full = run_two(masks, blend, false, imgs);
  const cv::Mat mini = run_two(masks, blend, true, imgs);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// The seam maximum is not an upper bound on where the band reaches, so the ROI pads from the
// request. A port of the N fixture: canvas height matters, because at 200 rows the coverage
// distance saturates and the cap pins the radius below overlap_padding whatever the fraction is.
TEST(CudaPanoAlphaTest, MinimizeBlendMatchesFullCanvasWhenTheSeamIsPinchedButTheBandIsNot) {
  constexpr int canvas_w = 1200, h = 900, seam_x = 600, seam_w = 5;
  // Image 1 blankets; image 2 is a narrower window whose left edge sits just left of the seam.
  cv::Mat seam(h, canvas_w, CV_8U, cv::Scalar(1));
  seam.colRange(seam_x, seam_x + seam_w).setTo(0);
  ControlMasks masks;
  masks.img1_col = make_identity_map_x(canvas_w, h);
  masks.img1_row = make_identity_map_y(canvas_w, h);
  masks.img2_col = make_identity_map_x(605, h);
  masks.img2_row = make_identity_map_y(605, h);
  masks.whole_seam_mask_image = seam;
  masks.positions = {SpatialTiff{0.0f, 0.0f}, SpatialTiff{595.0f, 0.0f}};
  ASSERT_TRUE(masks.is_valid());
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(canvas_w, h, 0), make_pattern_image_f4(605, h, 1)};

  // narrowest = 605, so 0.9 asks for 544.5 and max_px clamps it to 512 against a 22 px seam.
  const BlendSettings blend = BlendSettings::Alpha(0.9f);
  {
    hm::pano::cuda::CudaStitchPano<float4, float4> probe(1, blend, masks, true, true);
    ASSERT_TRUE(probe.status().ok()) << probe.status().message();
    ASSERT_TRUE(probe.minimizes_blend()) << "fixture too small for the blend ROI to engage";
  }
  const cv::Mat full = run_two(masks, blend, false, imgs);
  const cv::Mat mini = run_two(masks, blend, true, imgs);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// The guard has to read the labels handed in, not the corrected ones. A hole flush to the canvas
// edge leaves part of itself outside even the corrected-label ROI, so a guard reading corrected
// labels passes there (the correction found a covering owner) while the hard baseline it protects
// still has no data.
TEST(CudaPanoAlphaTest, MinimizeBlendGuardReadsTheLabelsTheHardBaselineWasBuiltFrom) {
  constexpr int w = 3400, h = 400, seam_x = 1700;
  cv::Mat seam(h, w, CV_8U, cv::Scalar(0));
  seam.colRange(0, seam_x).setTo(1);
  ControlMasks masks = make_masks(w, h, 0, seam);
  // Label 0 owns x >= 1700 and is image 2. The hole runs to the right canvas edge.
  masks.img2_col(cv::Rect(w - 400, 0, 400, h)).setTo(65535);
  masks.img2_row(cv::Rect(w - 400, 0, 400, h)).setTo(65535);
  const std::vector<cv::Mat> imgs = {make_pattern_image_f4(w, h, 0), make_pattern_image_f4(w, h, 1)};

  const BlendSettings blend = BlendSettings::Alpha(0.05f);
  const cv::Mat full = run_two(masks, blend, false, imgs);
  const cv::Mat mini = run_two(masks, blend, true, imgs);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini.empty());
  expect_mats_near(full, mini, kTol);
}

// Alpha mode builds no Laplacian context, so the managed-output path must not reach for one.
TEST(CudaPanoAlphaTest, CompactManagedOutputWorks) {
  constexpr int w = 97, h = 35, stride = 24;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 60);
  hm::pano::cuda::CudaStitchPano<float4, float4> reference(
      1, BlendSettings::Alpha(0.12f), masks, true, false, 0, false);
  hm::pano::cuda::CudaStitchPano<float4, float4> compact(1, BlendSettings::Alpha(0.12f), masks, true, false, 0, true);
  ASSERT_TRUE(reference.status().ok()) << reference.status().message();
  ASSERT_TRUE(compact.status().ok()) << compact.status().message();
  EXPECT_GT(reference.feather_radius_px(), 0.0f);

  auto a = upload<float4>(make_pattern_image_f4(w, h, 0));
  auto b = upload<float4>(make_pattern_image_f4(w, h, 1));
  auto expected = reference.process(*a, *b, 0, nullptr);
  ASSERT_TRUE(expected.ok()) << expected.status().message();
  auto actual = compact.process(*a, *b, 0, nullptr);
  ASSERT_TRUE(actual.ok()) << actual.status().message();
  CUDA_CHECK(cudaDeviceSynchronize());
  cv::Mat owned = expected.ConsumeValueOrDie()->download();
  cv::Mat borrowed = actual.ConsumeValueOrDie()->download();
  ASSERT_EQ(owned.total() * owned.elemSize(), borrowed.total() * borrowed.elemSize());
  EXPECT_EQ(std::memcmp(owned.data, borrowed.data, owned.total() * owned.elemSize()), 0);
}

// The pyramid dump is Laplacian-only and must refuse rather than dereference a missing context.
TEST(CudaPanoAlphaTest, PyramidDumpRejectsAlphaMode) {
  constexpr int w = 64, h = 24, stride = 16;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 40);
  hm::pano::cuda::CudaStitchPano<float4, float4> pano(1, BlendSettings::Alpha(0.1f), masks, true, false, 0, false);
  ASSERT_TRUE(pano.status().ok()) << pano.status().message();
  EXPECT_FALSE(pano.dump_soft_blend_pyramid("/unused-alpha-mode", /*stream=*/0).ok());
}

// hstream's fp16 path is CudaStitchPano<uchar4, half3>: three-channel compute, where the blend
// kernel does not drop zero-alpha contributors, so nothing but the weights stops a camera
// contributing outside its own footprint. It does not catch a field that fails to sum to one -
// the two-image kernel takes a single-channel mask and synthesizes 1-m, so the output is a convex
// combination whatever the host field summed to; normalization is pinned in featherMask_test
// instead. Nor does it catch black leakage: the seam ramp saturates well inside both footprints
// here, so nothing reaches an edge. What it pins is that the fp16 path runs at all and returns
// the constant it was given.
TEST(CudaPanoAlphaTest, ThreeChannelComputePreservesConstantInput) {
  constexpr int w = 97, h = 35, stride = 24;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 60);
  const cv::Vec4b colour(60, 120, 200, 255);
  cv::Mat host(h, w, CV_8UC4, colour);

  hm::pano::cuda::CudaStitchPano<uchar4, half3> pano(
      1, BlendSettings::Alpha(0.2f), masks, /*quiet=*/true, /*minimize_blend=*/false);
  ASSERT_TRUE(pano.status().ok()) << pano.status().message();

  hm::CudaMat<uchar4> a(host), b(host);
  auto out = pano.process(a, b, 0, std::make_unique<hm::CudaMat<uchar4>>(1, pano.canvas_width(), pano.canvas_height()));
  ASSERT_TRUE(out.ok()) << out.status().message();
  CUDA_CHECK(cudaDeviceSynchronize());
  const cv::Mat result = out.ConsumeValueOrDie()->download();
  ASSERT_EQ(result.type(), CV_8UC4);

  // Every pixel either carries the constant or is outside both footprints.
  const cv::Rect covered(0, 0, w + stride, h);
  for (int y = 0; y < result.rows; ++y) {
    const cv::Vec4b* row = result.ptr<cv::Vec4b>(y);
    for (int x = covered.x; x < covered.x + covered.width; ++x) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_NEAR(row[x][c], colour[c], 2) << "(" << x << "," << y << ") ch " << c;
      }
    }
  }
}

TEST(CudaPanoAlphaTest, RejectsOutOfRangeFeather) {
  constexpr int w = 64, h = 16, stride = 16;
  const ControlMasks masks = make_overlap_masks(w, h, stride, 40);
  hm::pano::cuda::CudaStitchPano<float4, float4> pano(1, BlendSettings::Alpha(-0.5f), masks, true, false, 0, false);
  EXPECT_FALSE(pano.status().ok());
}
