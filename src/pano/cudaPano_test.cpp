#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>

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
      /*match_exposure=*/false,
      /*quiet=*/true,
      /*minimize_blend=*/false);
  hm::pano::cuda::CudaStitchPano<float4, float4> pano_mini(
      /*batch_size=*/1,
      /*num_levels=*/kLevels,
      masks,
      /*match_exposure=*/false,
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
