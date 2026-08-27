// cudaPano3_test.cpp

#include <cupano/gpu/gpu_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

#include <array>
#include <cmath>
#include <memory>

// Include your project headers; adjust include paths as needed:
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/controlMasksN.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/pano/cudaPanoN.h"
#include "cupano/pano/cvTypes.h"

using ControlMasks3 = hm::pano::ControlMasks3;
using SpatialTiff = hm::pano::SpatialTiff;

using namespace hm;

// ----------------------------------------------------------------------------
// Utility macro to check CUDA calls in tests and fail on error.
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                             \
  do {                                                                                               \
    cudaError_t err = (call);                                                                        \
    if (err != cudaSuccess) {                                                                        \
      FAIL() << "CUDA error at " << __FILE__ << ":" << __LINE__ << " – " << cudaGetErrorString(err); \
    }                                                                                                \
  } while (0)

namespace {

cv::Mat make_identity_map_x(int width, int height) {
  cv::Mat map(height, width, CV_16U);
  for (int y = 0; y < height; ++y) {
    uint16_t* row = map.ptr<uint16_t>(y);
    for (int x = 0; x < width; ++x)
      row[x] = static_cast<uint16_t>(x);
  }
  return map;
}

cv::Mat make_identity_map_y(int width, int height) {
  cv::Mat map(height, width, CV_16U);
  for (int y = 0; y < height; ++y) {
    uint16_t* row = map.ptr<uint16_t>(y);
    for (int x = 0; x < width; ++x)
      row[x] = static_cast<uint16_t>(y);
  }
  return map;
}

cv::Mat make_pattern_image(int width, int height, int image_index) {
  cv::Mat image(height, width, CV_32FC4);
  for (int y = 0; y < height; ++y) {
    cv::Vec4f* row = image.ptr<cv::Vec4f>(y);
    for (int x = 0; x < width; ++x) {
      const float base = static_cast<float>(image_index * 50);
      row[x] = cv::Vec4f(
          base + static_cast<float>(x % 17),
          base + static_cast<float>(y % 19),
          base + static_cast<float>((x + y) % 23),
          255.0f);
    }
  }
  return image;
}

ControlMasks3 make_masks3(
    const std::array<cv::Size, 3>& sizes,
    const std::array<cv::Point, 3>& positions,
    const cv::Mat& seam_index) {
  ControlMasks3 masks;
  masks.img0_col = make_identity_map_x(sizes[0].width, sizes[0].height);
  masks.img0_row = make_identity_map_y(sizes[0].width, sizes[0].height);
  masks.img1_col = make_identity_map_x(sizes[1].width, sizes[1].height);
  masks.img1_row = make_identity_map_y(sizes[1].width, sizes[1].height);
  masks.img2_col = make_identity_map_x(sizes[2].width, sizes[2].height);
  masks.img2_row = make_identity_map_y(sizes[2].width, sizes[2].height);
  masks.whole_seam_mask_image = seam_index.clone();
  masks.positions = {
      SpatialTiff{static_cast<float>(positions[0].x), static_cast<float>(positions[0].y)},
      SpatialTiff{static_cast<float>(positions[1].x), static_cast<float>(positions[1].y)},
      SpatialTiff{static_cast<float>(positions[2].x), static_cast<float>(positions[2].y)}};
  EXPECT_TRUE(masks.is_valid());
  return masks;
}

cv::Mat run_pano3(
    const ControlMasks3& masks,
    const std::array<cv::Mat, 3>& host_inputs,
    int num_levels,
    bool minimize_blend,
    bool fused,
    int max_output_width = 0) {
  std::array<std::unique_ptr<hm::CudaMat<float4>>, 3> inputs = {
      std::make_unique<hm::CudaMat<float4>>(host_inputs[0]),
      std::make_unique<hm::CudaMat<float4>>(host_inputs[1]),
      std::make_unique<hm::CudaMat<float4>>(host_inputs[2])};
  hm::pano::cuda::CudaStitchPano3<float4, float4> pano(
      /*batch_size=*/1,
      num_levels,
      masks,
      /*quiet=*/true,
      max_output_width,
      minimize_blend);
  if (!pano.status().ok()) {
    ADD_FAILURE() << pano.status().message();
    return {};
  }

  cv::Mat stale_canvas(pano.canvas_height(), pano.canvas_width(), CV_32FC4, cv::Scalar(7, 11, 13, 17));
  auto canvas = std::make_unique<hm::CudaMat<float4>>(stale_canvas);
  auto result = pano.process(*inputs[0], *inputs[1], *inputs[2], /*stream=*/0, std::move(canvas), fused);
  if (!result.ok()) {
    ADD_FAILURE() << result.status().message();
    return {};
  }
  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    ADD_FAILURE() << "CUDA error: " << cudaGetErrorString(sync_status);
    return {};
  }
  return result.ConsumeValueOrDie()->download();
}

void expect_mats_near(const cv::Mat& expected, const cv::Mat& actual, float tolerance = 1e-4f) {
  ASSERT_EQ(expected.size(), actual.size());
  ASSERT_EQ(expected.type(), actual.type());
  double max_diff = 0.0;
  cv::Point max_location;
  cv::minMaxLoc(cv::abs(expected.reshape(1) - actual.reshape(1)), nullptr, &max_diff, nullptr, &max_location);
  EXPECT_LE(max_diff, tolerance) << " at (" << max_location.x / expected.channels() << ", " << max_location.y
                                 << ") channel " << max_location.x % expected.channels();
}

} // namespace

TEST(BlendRoiTest, FindsHorizontalAndVerticalOneDimensionalBoundaries) {
  cv::Mat horizontal(1, 7, CV_8U, cv::Scalar(0));
  horizontal.colRange(3, 7).setTo(1);
  const auto horizontal_bbox = hm::pano::blend_roi::seam_boundary_bbox(horizontal);
  ASSERT_TRUE(horizontal_bbox.has_value());
  EXPECT_EQ(*horizontal_bbox, cv::Rect(2, 0, 2, 1));

  cv::Mat vertical(7, 1, CV_8U, cv::Scalar(0));
  vertical.rowRange(4, 7).setTo(2);
  const auto vertical_bbox = hm::pano::blend_roi::seam_boundary_bbox(vertical);
  ASSERT_TRUE(vertical_bbox.has_value());
  EXPECT_EQ(*vertical_bbox, cv::Rect(0, 3, 1, 2));
}

TEST(BlendRoiTest, UnionsDisconnectedBoundariesAndAlignsBlendOrigin) {
  cv::Mat seam(384, 768, CV_8U, cv::Scalar(0));
  seam.colRange(256, 512).setTo(1);
  seam.colRange(512, 768).setTo(2);

  const hm::pano::blend_roi::Regions regions = hm::pano::blend_roi::select_regions(seam, 4, 128);
  ASSERT_GT(regions.blend.area(), 0);
  EXPECT_EQ(regions.blend.x % hm::pano::blend_roi::pyramid_alignment(4), 0);
  EXPECT_EQ(regions.blend.y % hm::pano::blend_roi::pyramid_alignment(4), 0);
  EXPECT_LT(regions.blend.area(), seam.cols * seam.rows);
  EXPECT_LE(regions.blend.x, regions.write.x);
  EXPECT_GE(regions.blend.x + regions.blend.width, regions.write.x + regions.write.width);
}

TEST(BlendRoiTest, ConstantMaskFallsBackToFullBlend) {
  cv::Mat seam(33, 65, CV_8U, cv::Scalar(1));
  const hm::pano::blend_roi::Regions regions = hm::pano::blend_roi::select_regions(seam, 4, 128);
  EXPECT_EQ(regions.blend.area(), 0);
  EXPECT_EQ(regions.write.area(), 0);
}

TEST(BlendRoiTest, NearFullBlendRoiFallsBack) {
  cv::Mat seam(256, 1024, CV_8U, cv::Scalar(0));
  seam.colRange(170, 830).setTo(1);
  const hm::pano::blend_roi::Regions regions = hm::pano::blend_roi::select_regions(seam, 4, 128);
  EXPECT_EQ(regions.blend.area(), 0);
  EXPECT_EQ(regions.write.area(), 0);
}

// ----------------------------------------------------------------------------
// 1) Invalid ControlMasks3: constructor should set an error status.
// ----------------------------------------------------------------------------

#if 0
TEST(CudaStitchPano3_InvalidControlMasks, ConstructorReportsError) {
  // Create an empty ControlMasks3 (never loaded), so is_valid() == false.
  ControlMasks3 badMasks;
  ASSERT_FALSE(badMasks.is_valid());

  // Construct the 3‐image stitcher with invalid masks:
  hm::pano::cuda::CudaStitchPano3<unsigned char, float> stitch(
      /*batch_size=*/1,
      /*num_levels=*/1,
      badMasks,
      /*quiet=*/true);

  // The internal status should be an error.
  auto status = stitch.status();
  ASSERT_FALSE(status.ok());
}

// ----------------------------------------------------------------------------
// 2) Hard‐seam trivial: a 1×1 mask value = 2 → third image should appear.
// ----------------------------------------------------------------------------
TEST(CudaStitchPano3_HardSeamTrivial, ThirdImageWins) {
  // 2a) Build a 1×1 single‐channel uchar mask with value 2 (indicating “use image #3”).
  cv::Mat seam_mask(1, 1, CV_8U, cv::Scalar(2));

  // 2b) Build 1×1 remap mats for all three images (identity mappings).
  cv::Mat map1x(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map1y(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map2x(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map2y(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map3x(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map3y(1, 1, CV_16U, cv::Scalar(0));

  // 2c) Define three spatial positions so that all three images map to (0,0) in a 1×1 canvas.
  SpatialTiff pos0{0.0f, 0.0f}, pos1{0.0f, 0.0f}, pos2{0.0f, 0.0f};
  std::vector<SpatialTiff> positions = {pos0, pos1, pos2};

  // 2d) Fill a valid ControlMasks3 with all six remap mats, the seam mask, and three positions.
  ControlMasks3 masks;
  masks.img0_col = map1x;
  masks.img0_row = map1y;
  masks.img1_col = map2x;
  masks.img1_row = map2y;
  masks.img2_col = map3x;
  masks.img2_row = map3y;
  masks.whole_seam_mask_image = seam_mask;
  masks.positions = positions;
  ASSERT_TRUE(masks.is_valid());
  ASSERT_EQ(masks.canvas_width(), 1u);
  ASSERT_EQ(masks.canvas_height(), 1u);

  // 2e) Create three 1×1 host images (uchar):
  //     image1 pixel=10, image2 pixel=50, image3 pixel=200.
  cv::Mat host1(1, 1, CV_8UC3, cv::Scalar(10));
  cv::Mat host2(1, 1, CV_8UC3, cv::Scalar(50));
  cv::Mat host3(1, 1, CV_8UC3, cv::Scalar(200));

  // 2f) Upload them into CudaMat<uchar>:
  using CudaMatU = hm::CudaMat<uchar3>;
  auto d_img0 = std::make_unique<CudaMatU>(/*batchSize=*/1, /*width=*/1, /*height=*/1);
  auto d_img1 = std::make_unique<CudaMatU>(/*batchSize=*/1, /*width=*/1, /*height=*/1);
  auto d_img2 = std::make_unique<CudaMatU>(/*batchSize=*/1, /*width=*/1, /*height=*/1);
  CUDA_CHECK(d_img0->upload(host1));
  CUDA_CHECK(d_img1->upload(host2));
  CUDA_CHECK(d_img2->upload(host3));

  // 2g) Create a 1×1 output canvas:
  auto d_canvas = std::make_unique<CudaMatU>(/*batchSize=*/1, /*width=*/1, /*height=*/1);

  // 2h) Instantiate the 3-image stitcher in hard‐seam mode (num_levels=0):
  hm::pano::cuda::CudaStitchPano3<uchar3, float3> stitch(
      /*batch_size=*/1,
      /*num_levels=*/0,
      masks,
      /*quiet=*/true);

  ASSERT_TRUE(stitch.status().ok());

  // 2i) Call process(): expect the single pixel to come from image3 (value=200).
  auto resultOr = stitch.process(*d_img0, *d_img1, *d_img2, /*stream=*/0, std::move(d_canvas));
  ASSERT_TRUE(resultOr.ok()) << resultOr.status().message();

  std::unique_ptr<CudaMatU> d_out = std::move(resultOr.ConsumeValueOrDie());
  ASSERT_EQ(d_out->width(), 1);
  ASSERT_EQ(d_out->height(), 1);

  // 2j) Download and verify:
  cv::Mat hostOut = d_out->download();
  ASSERT_EQ(hostOut.type(), CV_8UC3);
  uchar pixel = hostOut.at<uchar>(0, 0);
  EXPECT_EQ(pixel, static_cast<uchar>(200));
}
#endif

// ----------------------------------------------------------------------------
// 3) Soft‐seam trivial: 1×1 labeled seam with value=1 (one-hot for image #1).
//    Expect the output pixel to match image #1 exactly.
// ----------------------------------------------------------------------------
TEST(CudaStitchPano3_SoftSeamTrivial, OneHotLabelSelectsMiddleImage) {
  // 3a) Create a 1×1 indexed seam (CV_8U) with value=1 → selects image #1.
  cv::Mat seam_mask_f(1, 1, CV_8U, cv::Scalar(1));

  // All map to same pixel 0,0
  // 3b) Remap mats same as before:
  cv::Mat map1x(1, 1, CV_16U, cv::Scalar(0)), map1y(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map2x(1, 1, CV_16U, cv::Scalar(0)), map2y(1, 1, CV_16U, cv::Scalar(0));
  cv::Mat map3x(1, 1, CV_16U, cv::Scalar(0)), map3y(1, 1, CV_16U, cv::Scalar(0));

  // All in same place
  SpatialTiff pos0{0.0f, 0.0f}, pos1{0.0f, 0.0f}, pos2{0.0f, 0.0f};
  std::vector<SpatialTiff> positions = {pos0, pos1, pos2};

  ControlMasks3 masks;
  masks.img0_col = map1x;
  masks.img0_row = map1y;
  masks.img1_col = map2x;
  masks.img1_row = map2y;
  masks.img2_col = map3x;
  masks.img2_row = map3y;
  masks.whole_seam_mask_image = seam_mask_f;
  masks.positions = positions;
  ASSERT_TRUE(masks.is_valid());

  // 3c) Create three 1×1 float images: image1=30.0, image2=60.0, image3=90.0
  using CudaMatF = hm::CudaMat<float4>;
  cv::Mat host1(1, 1, CV_32FC4, cv::Scalar(30.0f, 30.0f, 30.0f, 255.0));
  cv::Mat host2(1, 1, CV_32FC4, cv::Scalar(60.0f, 60.0f, 60.0f, 255.0));
  cv::Mat host3(1, 1, CV_32FC4, cv::Scalar(90.0f, 90.0f, 90.0f, 255.0));
  auto d_img1 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img2 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img3 = std::make_unique<CudaMatF>(1, 1, 1);
  CUDA_CHECK(d_img1->upload(host1));
  CUDA_CHECK(d_img2->upload(host2));
  CUDA_CHECK(d_img3->upload(host3));

  auto d_canvas = std::make_unique<CudaMatF>(1, 1, 1);

  // 3d) Instantiate with num_levels=1 (soft seam)
  hm::pano::cuda::CudaStitchPano3<float4, float4> stitch(
      /*batch_size=*/1,
      /*num_levels=*/1,
      masks,
      /*quiet=*/true);
  ASSERT_TRUE(stitch.status().ok());

  // 3e) Call process(). Expect output equals image #1 value (60).
  auto resultOr = stitch.process(*d_img1, *d_img2, *d_img3, /*stream=*/0, std::move(d_canvas));
  ASSERT_TRUE(resultOr.ok()) << resultOr.status().message();

  std::unique_ptr<CudaMatF> d_out = std::move(resultOr).ConsumeValueOrDie();
  ASSERT_EQ(d_out->width(), 1);
  ASSERT_EQ(d_out->height(), 1);

  cv::Mat hostOut = d_out->download();
  ASSERT_EQ(hostOut.type(), CV_32FC4);
  cv::Vec4f pixel = hostOut.at<cv::Vec4f>(0, 0);
  EXPECT_NEAR(pixel[0], 60.0f, 1e-3f);
  EXPECT_NEAR(pixel[1], 60.0f, 1e-3f);
  EXPECT_NEAR(pixel[2], 60.0f, 1e-3f);
  // Preserve alpha
  EXPECT_NEAR(pixel[3], 255.0f, 1e-3f);
}

TEST(CudaStitchPano3_MaxOutputWidth, ConstructorScalesMasksBeforeCanvasAllocation) {
  constexpr int W = 64;
  constexpr int H = 32;
  cv::Mat seam_mask(H, 160, CV_8U, cv::Scalar(0));
  seam_mask.colRange(48, 96).setTo(1);
  seam_mask.colRange(96, 160).setTo(2);

  ControlMasks3 masks;
  masks.img0_col = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.img0_row = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.img1_col = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.img1_row = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.img2_col = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.img2_row = cv::Mat(H, W, CV_16U, cv::Scalar(0));
  masks.whole_seam_mask_image = seam_mask;
  masks.positions = {SpatialTiff{0.0f, 0.0f}, SpatialTiff{48.0f, 0.0f}, SpatialTiff{96.0f, 0.0f}};
  ASSERT_TRUE(masks.is_valid());

  hm::pano::cuda::CudaStitchPano3<float4, float4> stitch(
      /*batch_size=*/1,
      /*num_levels=*/0,
      masks,
      /*quiet=*/true,
      /*max_output_width=*/80);

  ASSERT_TRUE(stitch.status().ok()) << stitch.status().message();
  EXPECT_EQ(stitch.canvas_width(), 80);
  EXPECT_EQ(stitch.canvas_height(), 16);
  EXPECT_EQ(masks.canvas_width(), 160u);
  EXPECT_EQ(masks.canvas_height(), 32u);
}

TEST(CudaStitchPano3_MinimizeBlend, FusedAndLegacyMatchFullCanvasAcrossSeparatedSeams) {
  constexpr int W = 768;
  constexpr int H = 384;
  constexpr int LEVELS = 4;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 3, 2 * W / 3).setTo(1);
  seam.colRange(2 * W / 3, W).setTo(2);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  hm::pano::cuda::CudaStitchPano3<float4, float4> minimized(
      /*batch_size=*/1,
      LEVELS,
      masks,
      /*quiet=*/true,
      /*max_output_width=*/0,
      /*minimize_blend=*/true);
  ASSERT_TRUE(minimized.status().ok()) << minimized.status().message();
  ASSERT_TRUE(minimized.minimizes_blend());
  EXPECT_LT(minimized.blend_roi_canvas().area(), W * H);

  const cv::Mat full_fused = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/true);
  const cv::Mat minimized_fused = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/true);
  const cv::Mat full_legacy = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/false);
  const cv::Mat minimized_legacy = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/false);

  ASSERT_FALSE(full_fused.empty());
  ASSERT_FALSE(minimized_fused.empty());
  ASSERT_FALSE(full_legacy.empty());
  ASSERT_FALSE(minimized_legacy.empty());
  expect_mats_near(full_fused, minimized_fused);
  expect_mats_near(full_fused, full_legacy);
  expect_mats_near(full_fused, minimized_legacy);
}

TEST(CudaStitchPano3_MinimizeBlend, NonzeroVerticalBlendOriginMatchesFullCanvas) {
  constexpr int W = 384;
  constexpr int H = 1024;
  constexpr int LEVELS = 3;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam(cv::Rect(W / 3, 3 * H / 8, W / 3, H / 4)).setTo(1);
  seam(cv::Rect(2 * W / 3, 3 * H / 8, W / 3, H / 4)).setTo(2);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  hm::pano::cuda::CudaStitchPano3<float4, float4> minimized(
      /*batch_size=*/1,
      LEVELS,
      masks,
      /*quiet=*/true,
      /*max_output_width=*/0,
      /*minimize_blend=*/true);
  ASSERT_TRUE(minimized.status().ok()) << minimized.status().message();
  ASSERT_TRUE(minimized.minimizes_blend());
  ASSERT_GT(minimized.blend_roi_canvas().y, 0);
  EXPECT_LT(minimized.blend_roi_canvas().area(), W * H);

  const cv::Mat full_fused = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/true);
  const cv::Mat minimized_fused = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/true);
  const cv::Mat full_legacy = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/false);
  const cv::Mat minimized_legacy = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/false);

  ASSERT_FALSE(full_fused.empty());
  ASSERT_FALSE(minimized_fused.empty());
  ASSERT_FALSE(full_legacy.empty());
  ASSERT_FALSE(minimized_legacy.empty());
  expect_mats_near(full_fused, minimized_fused);
  expect_mats_near(full_fused, full_legacy);
  expect_mats_near(full_fused, minimized_legacy);
}

TEST(CudaStitchPano3_MinimizeBlend, CappedAndUnmappedSoftSeamsMatchFullCanvas) {
  constexpr int W = 768;
  constexpr int H = 384;
  constexpr int LEVELS = 4;
  constexpr int MAX_OUTPUT_WIDTH = W / 2;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 3, 2 * W / 3).setTo(1);
  seam.colRange(2 * W / 3, W).setTo(2);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  constexpr uint16_t UNMAPPED = 65535;
  masks.img1_col(cv::Rect(W / 2 - 8, H / 2 - 8, 16, 16)).setTo(UNMAPPED);
  masks.img1_row(cv::Rect(W / 2 - 8, H / 2 - 8, 16, 16)).setTo(UNMAPPED);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  hm::pano::cuda::CudaStitchPano3<float4, float4> minimized(
      /*batch_size=*/1,
      LEVELS,
      masks,
      /*quiet=*/true,
      MAX_OUTPUT_WIDTH,
      /*minimize_blend=*/true);
  ASSERT_TRUE(minimized.status().ok()) << minimized.status().message();
  ASSERT_TRUE(minimized.minimizes_blend());
  EXPECT_EQ(minimized.canvas_width(), MAX_OUTPUT_WIDTH);
  EXPECT_LT(minimized.blend_roi_canvas().area(), minimized.canvas_width() * minimized.canvas_height());

  const cv::Mat full = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/true, MAX_OUTPUT_WIDTH);
  const cv::Mat mini_fused =
      run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/true, MAX_OUTPUT_WIDTH);
  const cv::Mat mini_legacy =
      run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/false, MAX_OUTPUT_WIDTH);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(mini_fused.empty());
  ASSERT_FALSE(mini_legacy.empty());
  expect_mats_near(full, mini_fused);
  expect_mats_near(full, mini_legacy);
}

TEST(CudaStitchPano3_MinimizeBlend, SpecializedThreeMatchesGenericNThree) {
  constexpr int W = 768;
  constexpr int H = 256;
  constexpr int LEVELS = 4;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 3, 2 * W / 3).setTo(1);
  seam.colRange(2 * W / 3, W).setTo(2);
  ControlMasks3 masks3 = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> host_inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};
  const cv::Mat specialized = run_pano3(masks3, host_inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/true);
  ASSERT_FALSE(specialized.empty());

  hm::pano::ControlMasksN masks_n;
  masks_n.img_col = {masks3.img0_col, masks3.img1_col, masks3.img2_col};
  masks_n.img_row = {masks3.img0_row, masks3.img1_row, masks3.img2_row};
  masks_n.whole_seam_mask_indexed = seam.clone();
  masks_n.positions = masks3.positions;
  ASSERT_TRUE(masks_n.is_valid());

  std::array<std::unique_ptr<hm::CudaMat<float4>>, 3> inputs = {
      std::make_unique<hm::CudaMat<float4>>(host_inputs[0]),
      std::make_unique<hm::CudaMat<float4>>(host_inputs[1]),
      std::make_unique<hm::CudaMat<float4>>(host_inputs[2])};
  const std::vector<const hm::CudaMat<float4>*> input_ptrs = {inputs[0].get(), inputs[1].get(), inputs[2].get()};
  hm::pano::cuda::CudaStitchPanoN<float4, float4> generic(
      /*batch_size=*/1,
      LEVELS,
      masks_n,
      /*minimize_blend=*/true,
      /*quiet=*/true);
  ASSERT_TRUE(generic.status().ok()) << generic.status().message();
  ASSERT_TRUE(generic.minimizes_blend());
  auto canvas = std::make_unique<hm::CudaMat<float4>>(1, generic.canvas_width(), generic.canvas_height());
  auto generic_result = generic.process(input_ptrs, /*stream=*/0, std::move(canvas));
  ASSERT_TRUE(generic_result.ok()) << generic_result.status().message();
  CUDA_CHECK(cudaDeviceSynchronize());
  const cv::Mat generic_output = generic_result.ConsumeValueOrDie()->download();
  expect_mats_near(specialized, generic_output);
}

TEST(CudaStitchPano3_MinimizeBlend, ReorderedOffsetFootprintsMatchFullCanvas) {
  constexpr int W = 384;
  constexpr int H = 320;
  constexpr int CANVAS_W = 640;
  constexpr int CANVAS_H = 416;
  constexpr int LEVELS = 3;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(256, 96), cv::Point(0, 0), cv::Point(128, 48)};
  cv::Mat seam(CANVAS_H, CANVAS_W, CV_8U, cv::Scalar(1));
  seam.colRange(224, 416).setTo(2);
  seam.colRange(416, CANVAS_W).setTo(0);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  const cv::Mat full = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/false, /*fused=*/true);
  const cv::Mat minimized = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/true);
  const cv::Mat minimized_legacy = run_pano3(masks, inputs, LEVELS, /*minimize_blend=*/true, /*fused=*/false);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(minimized.empty());
  ASSERT_FALSE(minimized_legacy.empty());
  expect_mats_near(full, minimized);
  expect_mats_near(full, minimized_legacy);
}

TEST(CudaStitchPano3_MinimizeBlend, NoBoundaryUsesFullSoftFallback) {
  constexpr int W = 96;
  constexpr int H = 64;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(1));
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  hm::pano::cuda::CudaStitchPano3<float4, float4> pano(
      /*batch_size=*/1,
      /*num_levels=*/3,
      masks,
      /*quiet=*/true,
      /*max_output_width=*/0,
      /*minimize_blend=*/true);
  ASSERT_TRUE(pano.status().ok()) << pano.status().message();
  EXPECT_FALSE(pano.minimizes_blend());

  const cv::Mat full = run_pano3(masks, inputs, /*num_levels=*/3, /*minimize_blend=*/false, /*fused=*/true);
  const cv::Mat fallback = run_pano3(masks, inputs, /*num_levels=*/3, /*minimize_blend=*/true, /*fused=*/true);
  ASSERT_FALSE(full.empty());
  ASSERT_FALSE(fallback.empty());
  expect_mats_near(full, fallback);
}

TEST(CudaStitchPano3_MinimizeBlend, HardSeamIgnoresMinimizeRequest) {
  constexpr int W = 64;
  constexpr int H = 32;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 3, 2 * W / 3).setTo(1);
  seam.colRange(2 * W / 3, W).setTo(2);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);
  const std::array<cv::Mat, 3> inputs = {
      make_pattern_image(W, H, 0), make_pattern_image(W, H, 1), make_pattern_image(W, H, 2)};

  const cv::Mat fused = run_pano3(masks, inputs, /*num_levels=*/0, /*minimize_blend=*/true, /*fused=*/true);
  const cv::Mat legacy = run_pano3(masks, inputs, /*num_levels=*/0, /*minimize_blend=*/true, /*fused=*/false);
  ASSERT_FALSE(fused.empty());
  ASSERT_FALSE(legacy.empty());
  expect_mats_near(fused, legacy, /*tolerance=*/0.0f);
}

TEST(CudaStitchPano3_MinimizeBlend, BatchTwoUsesCallerStreamAcrossFrames) {
  constexpr int W = 768;
  constexpr int H = 192;
  constexpr int LEVELS = 3;
  const std::array<cv::Size, 3> sizes = {cv::Size(W, H), cv::Size(W, H), cv::Size(W, H)};
  const std::array<cv::Point, 3> positions = {cv::Point(0, 0), cv::Point(0, 0), cv::Point(0, 0)};
  cv::Mat seam(H, W, CV_8U, cv::Scalar(0));
  seam.colRange(W / 3, 2 * W / 3).setTo(1);
  seam.colRange(2 * W / 3, W).setTo(2);
  ControlMasks3 masks = make_masks3(sizes, positions, seam);

  std::array<std::vector<cv::Mat>, 3> host_batches;
  for (int image = 0; image < 3; ++image) {
    host_batches[image].push_back(make_pattern_image(W, H, image));
    cv::Mat changed = make_pattern_image(W, H, image);
    changed += cv::Scalar(3, 5, 7, 0);
    host_batches[image].push_back(changed);
  }
  std::array<hm::CudaMat<float4>, 3> inputs = {
      hm::CudaMat<float4>(host_batches[0]), hm::CudaMat<float4>(host_batches[1]), hm::CudaMat<float4>(host_batches[2])};

  hm::pano::cuda::CudaStitchPano3<float4, float4> full(
      /*batch_size=*/2, LEVELS, masks, /*quiet=*/true, /*max_output_width=*/0, /*minimize_blend=*/false);
  hm::pano::cuda::CudaStitchPano3<float4, float4> minimized(
      /*batch_size=*/2, LEVELS, masks, /*quiet=*/true, /*max_output_width=*/0, /*minimize_blend=*/true);
  ASSERT_TRUE(full.status().ok()) << full.status().message();
  ASSERT_TRUE(minimized.status().ok()) << minimized.status().message();
  ASSERT_TRUE(minimized.minimizes_blend());

  cudaStream_t stream{};
#if defined(USE_VULKAN)
  CUDA_CHECK(cudaStreamCreate(&stream));
#else
  CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
#endif
  auto full_canvas = std::make_unique<hm::CudaMat<float4>>(2, W, H);
  auto mini_canvas = std::make_unique<hm::CudaMat<float4>>(2, W, H);
  auto full_result = full.process(inputs[0], inputs[1], inputs[2], stream, std::move(full_canvas));
  ASSERT_TRUE(full_result.ok()) << full_result.status().message();
  auto mini_result = minimized.process(inputs[0], inputs[1], inputs[2], stream, std::move(mini_canvas));
  ASSERT_TRUE(mini_result.ok()) << mini_result.status().message();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto full_output = full_result.ConsumeValueOrDie();
  auto mini_output = mini_result.ConsumeValueOrDie();
  const cv::Mat first_full_0 = full_output->download(0);
  const cv::Mat first_full_1 = full_output->download(1);
  expect_mats_near(first_full_0, mini_output->download(0));
  expect_mats_near(first_full_1, mini_output->download(1));

  std::array<std::vector<cv::Mat>, 3> second_host_batches;
  for (int image = 0; image < 3; ++image) {
    for (int batch = 0; batch < 2; ++batch) {
      cv::Mat changed = make_pattern_image(W, H, image);
      changed += cv::Scalar(19 + batch, 23 + batch, 29 + batch, 0);
      second_host_batches[image].push_back(changed);
    }
  }
  std::array<hm::CudaMat<float4>, 3> second_inputs = {
      hm::CudaMat<float4>(second_host_batches[0]),
      hm::CudaMat<float4>(second_host_batches[1]),
      hm::CudaMat<float4>(second_host_batches[2])};
  const std::vector<cv::Mat> stale_full_batch(2, cv::Mat(H, W, CV_32FC4, cv::Scalar(31, 37, 41, 43)));
  const std::vector<cv::Mat> stale_mini_batch(2, cv::Mat(H, W, CV_32FC4, cv::Scalar(47, 53, 59, 61)));
  auto second_full_canvas = std::make_unique<hm::CudaMat<float4>>(stale_full_batch);
  auto second_mini_canvas = std::make_unique<hm::CudaMat<float4>>(stale_mini_batch);
  auto second_full_result =
      full.process(second_inputs[0], second_inputs[1], second_inputs[2], stream, std::move(second_full_canvas));
  ASSERT_TRUE(second_full_result.ok()) << second_full_result.status().message();
  auto second_mini_result =
      minimized.process(second_inputs[0], second_inputs[1], second_inputs[2], stream, std::move(second_mini_canvas));
  ASSERT_TRUE(second_mini_result.ok()) << second_mini_result.status().message();
  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto second_full_output = second_full_result.ConsumeValueOrDie();
  auto second_mini_output = second_mini_result.ConsumeValueOrDie();
  const cv::Mat second_full_0 = second_full_output->download(0);
  const cv::Mat second_full_1 = second_full_output->download(1);
  expect_mats_near(second_full_0, second_mini_output->download(0));
  expect_mats_near(second_full_1, second_mini_output->download(1));
  EXPECT_GT(cv::norm(first_full_0, second_full_0, cv::NORM_INF), 1.0);
  EXPECT_GT(cv::norm(first_full_1, second_full_1, cv::NORM_INF), 1.0);
  CUDA_CHECK(cudaStreamDestroy(stream));
}

#if 0
// Helper to compute expected average (integer truncation for uchar, exact for float)
template <typename T>
T expectedAverage(T a, T b, T c) {
  if constexpr (std::is_same_v<T, unsigned char>) {
    return static_cast<T>((int(a.x) + int(b.x) + int(c.x)) / 3);
  } else {
    return {
        static_cast<T>((a.x + b.x + c.x) / 3.0f),
        static_cast<T>((a.y + b.y + c.y) / 3.0f),
        static_cast<T>((a.y + b.y + c.y) / 3.0f)};
  }
}

// Test fixture for 3x3 blending with three separate masks
template <typename T = uchar3>
class Blend3x3Test : public ::testing::Test {
 protected:
  void SetUp() override {
    W = 3;
    H = 3;
    C = 1; // single‐channel
    B = 1; // batch size = 1
    numLevels = 1; // single‐level pyramid → no down/up sampling

    imageSize = W * H * C * B;
    maskSize = W * H * B;

    // Allocate host arrays
    h_im0.resize(imageSize);
    h_im1.resize(imageSize);
    h_im2.resize(imageSize);
    h_mask0.resize(maskSize);
    h_mask1.resize(maskSize);
    h_mask2.resize(maskSize);
    h_out.resize(imageSize);

    // Initialize three distinct 3×3 images:
    // image0 pixels = 0 + row*10 + col
    // image1 pixels = 100 + row*10 + col
    // image2 pixels = 200 + row*10 + col
    for (int r = 0; r < H; ++r) {
      for (int c = 0; c < W; ++c) {
        int idx = r * W + c;
        auto base = static_cast<T>(r * 10 + c);
        h_im0[idx] = base;
        h_im1[idx] = static_cast<T>(100 + (r * 10 + c));
        h_im2[idx] = static_cast<T>(200 + (r * 10 + c));
      }
    }

    // Zero‐initialize all masks
    std::fill(h_mask0.begin(), h_mask0.end(), 0.0f);
    std::fill(h_mask1.begin(), h_mask1.end(), 0.0f);
    std::fill(h_mask2.begin(), h_mask2.end(), 0.0f);

    // Set up mask so that:
    //  - At (0,0): mask0=1 (pure image0)
    //  - At (1,1): mask0=1, mask1=1, mask2=1 (equal blend)
    //  - At (2,2): mask2=1 (pure image2)
    auto setMask = [&](int r, int c, float m0, float m1, float m2) {
      int idx = r * W + c;
      h_mask0[idx] = m0;
      h_mask1[idx] = m1;
      h_mask2[idx] = m2;
    };
    setMask(0, 0, 1.0f, 0.0f, 0.0f);
    setMask(1, 1, 1.0f, 1.0f, 1.0f);
    setMask(2, 2, 0.0f, 0.0f, 1.0f);

    // The rest of the pixels keep masks = 0, so output should be 0 (from image0).

    // Allocate device arrays
    CUDA_CHECK(cudaMalloc((void**)&d_im0, imageSize * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_im1, imageSize * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_im2, imageSize * sizeof(T)));
    CUDA_CHECK(cudaMalloc((void**)&d_mask0, maskSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mask1, maskSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_mask2, maskSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_out, imageSize * sizeof(T)));

    // Copy host → device
    CUDA_CHECK(cudaMemcpy(d_im0, h_im0.data(), imageSize * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_im1, h_im1.data(), imageSize * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_im2, h_im2.data(), imageSize * sizeof(T), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask0, h_mask0.data(), maskSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask1, h_mask1.data(), maskSize * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask2, h_mask2.data(), maskSize * sizeof(float), cudaMemcpyHostToDevice));
  }

  void TearDown() override {
    cudaFree(d_im0);
    cudaFree(d_im1);
    cudaFree(d_im2);
    cudaFree(d_mask0);
    cudaFree(d_mask1);
    cudaFree(d_mask2);
    cudaFree(d_out);
  }

  int W, H, C, B, numLevels;
  int imageSize, maskSize;

  std::vector<T> h_im0, h_im1, h_im2, h_out;
  std::vector<float> h_mask0, h_mask1, h_mask2;

  T* d_im0;
  T* d_im1;
  T* d_im2;
  float* d_mask0;
  float* d_mask1;
  float* d_mask2;
  T* d_out;
};

TYPED_TEST_SUITE_P(Blend3x3Test);

// Test that the output at (0,0) comes purely from image0,
// at (1,1) is the average of image0/image1/image2,
// at (2,2) comes purely from image2. Other pixels default to image0 (since mask=0).
TYPED_TEST_P(Blend3x3Test, VerifyCornerAndCenterBlending) {
  using T = uchar3;

  // Invoke the no‐context 3‐image blender with one pyramid level:
  cudaError_t err = cudaBatchedLaplacianBlend3<uchar3, float3>(
      this->d_im0,
      this->d_im1,
      this->d_im2,
      this->d_mask0,
      this->d_mask1,
      this->d_mask2,
      this->d_out,
      this->W,
      this->H,
      this->C,
      this->numLevels,
      this->B,
      /*stream=*/0);
  ASSERT_EQ(err, cudaSuccess);

  // Copy device→host
  CUDA_CHECK(cudaMemcpy(this->h_out.data(), this->d_out, this->imageSize * sizeof(T), cudaMemcpyDeviceToHost));

  // (0,0): expect image0(0,0) = 0 + (0*10+0) = 0
  {
    int idx = 0 * this->W + 0;
    T expected = this->h_im0[idx];
    EXPECT_EQ(this->h_out[idx], expected) << "(0,0) expected " << int(expected) << ", got " << int(this->h_out[idx]);
  }
  // (1,1): average of image0(1,1)= (1*10+1)=11,
  //         image1(1,1)=100+(1*10+1)=111,
  //         image2(1,1)=200+(1*10+1)=211 → (11+111+211)/3 = 111
  {
    int idx = 1 * this->W + 1;
    T v0 = this->h_im0[idx];
    T v1 = this->h_im1[idx];
    T v2 = this->h_im2[idx];
    T expected = expectedAverage<T>(v0, v1, v2);
    EXPECT_EQ(this->h_out[idx], expected) << "(1,1) expected " << int(expected) << ", got " << int(this->h_out[idx]);
  }
  // (2,2): expect image2(2,2)=200+(2*10+2)=222
  {
    int idx = 2 * this->W + 2;
    T expected = this->h_im2[idx];
    EXPECT_EQ(this->h_out[idx], expected) << "(2,2) expected " << int(expected) << ", got " << int(this->h_out[idx]);
  }
  // Check another pixel, e.g. (0,1): mask0=mask1=mask2=0 → output should be image0(0,1)= (0*10+1)=1
  {
    int idx = 0 * this->W + 1;
    T expected = this->h_im0[idx];
    EXPECT_EQ(this->h_out[idx], expected) << "(0,1) expected " << int(expected) << ", got " << int(this->h_out[idx]);
  }
}

REGISTER_TYPED_TEST_SUITE_P(Blend3x3Test, VerifyCornerAndCenterBlending);

// Instantiate for unsigned char and float
using MyTypes = ::testing::Types<unsigned char, float>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Blend3x3Test, MyTypes);
#endif

// ----------------------------------------------------------------------------
// Main runner for Google Test
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
