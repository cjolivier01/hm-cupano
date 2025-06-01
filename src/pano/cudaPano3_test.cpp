// cudaPano3_test.cpp

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/core/hal/interface.h>

// Include your project headers; adjust include paths as needed:
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano3.h"
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

// ----------------------------------------------------------------------------
// 1) Invalid ControlMasks3: constructor should set an error status.
// ----------------------------------------------------------------------------
TEST(CudaStitchPano3_InvalidControlMasks, ConstructorReportsError) {
  // Create an empty ControlMasks3 (never loaded), so is_valid() == false.
  ControlMasks3 badMasks;
  ASSERT_FALSE(badMasks.is_valid());

  // Construct the 3‐image stitcher with invalid masks:
  hm::pano::cuda::CudaStitchPano3<unsigned char, float> stitch(
      /*batch_size=*/1,
      /*num_levels=*/1,
      badMasks,
      /*match_exposure=*/false,
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
  cv::Mat host1(1, 1, CV_8U, cv::Scalar(10));
  cv::Mat host2(1, 1, CV_8U, cv::Scalar(50));
  cv::Mat host3(1, 1, CV_8U, cv::Scalar(200));

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
      /*match_exposure=*/false,
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

#if 0

// ----------------------------------------------------------------------------
// 3) Soft‐seam trivial: single 1×1 float “label” mask = 1.5 (meaning equal weights
//    among all three?). For a 3-image float mask, assume soft‐seam logic interprets
//    any non-integer value as a uniform‐blend among all three. Then result = average.
// ----------------------------------------------------------------------------
TEST(CudaStitchPano3_SoftSeamTrivial, OutputEqualsTripleAverage) {
  // 3a) Create a 1×1 floating‐point mask. If the class expects a single float channel
  //     with range [0..2], we choose value=1.0 so that “image2” is fully selected? Or
  //     if it normalizes three masks internally, we can choose 0.5 to blend. Since the
  //     exact interpretation may vary, let’s choose 1.0 and assume “soft” means equal‐weight
  //     among all three. (Adjust if your implementation differs.)
  cv::Mat seam_mask_f(1, 1, CV_32F, cv::Scalar(1.0f));

  // 3b) Remap mats same as before:
  cv::Mat map1x(1, 1, CV_16U, cv::Scalar(0)), map1y(1, 1, CV_16U, cv::Scalar(0)), map2x(1, 1, CV_16U, cv::Scalar(0)),
      map2y(1, 1, CV_16U, cv::Scalar(0)), map3x(1, 1, CV_16U, cv::Scalar(0)), map3y(1, 1, CV_16U, cv::Scalar(0));
  SpatialTiff3 pos0{0.0f, 0.0f}, pos1{0.0f, 0.0f}, pos2{0.0f, 0.0f};
  std::vector<SpatialTiff3> positions = {pos0, pos1, pos2};

  ControlMasks3 masks;
  masks.img1_col = map1x;
  masks.img1_row = map1y;
  masks.img2_col = map2x;
  masks.img2_row = map2y;
  masks.img3_col = map3x;
  masks.img3_row = map3y;
  masks.whole_seam_mask_image = seam_mask_f;
  masks.positions = positions;
  ASSERT_TRUE(masks.is_valid());

  // 3c) Create three 1×1 float images: image1=30.0, image2=60.0, image3=90.0
  using CudaMatF = hm::CudaMat<float>;
  cv::Mat host1(1, 1, CV_32F, cv::Scalar(30.0f));
  cv::Mat host2(1, 1, CV_32F, cv::Scalar(60.0f));
  cv::Mat host3(1, 1, CV_32F, cv::Scalar(90.0f));
  auto d_img1 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img2 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img3 = std::make_unique<CudaMatF>(1, 1, 1);
  CUDA_CHECK(d_img1->upload(host1));
  CUDA_CHECK(d_img2->upload(host2));
  CUDA_CHECK(d_img3->upload(host3));

  auto d_canvas = std::make_unique<CudaMatF>(1, 1, 1);

  // 3d) Instantiate with num_levels=1 (soft seam)
  hm::pano::cuda::CudaStitchPano3<float, float> stitch(
      /*batch_size=*/1,
      /*num_levels=*/1,
      masks,
      /*match_exposure=*/false,
      /*quiet=*/true);
  ASSERT_TRUE(stitch.status().ok());

  // 3e) Call process(). For a simple “uniform blend” interpretation, the output = (30+60+90)/3 = 60.
  auto resultOr = stitch.process(*d_img1, *d_img2, *d_img3, /*stream=*/0, std::move(d_canvas));
  ASSERT_TRUE(resultOr.ok()) << resultOr.status().message();

  std::unique_ptr<CudaMatF> d_out = std::move(resultOr).value();
  ASSERT_EQ(d_out->width(), 1);
  ASSERT_EQ(d_out->height(), 1);

  cv::Mat hostOut;
  d_out->download(hostOut);
  ASSERT_EQ(hostOut.type(), CV_32F);
  float pixel = hostOut.at<float>(0, 0);
  EXPECT_NEAR(pixel, 60.0f, 1e-3f);
}

// ----------------------------------------------------------------------------
// 4) Invalid soft‐seam mask type: if num_levels>0 but mask is CV_8U, expect failure.
// ----------------------------------------------------------------------------
TEST(CudaStitchPano3_InvalidSoftMask, ProcessReportsError) {
  // Build a CV_8U mask but request num_levels=1 (soft seam), which expects CV_32F.
  cv::Mat seam_mask_u(1, 1, CV_8U, cv::Scalar(1));

  cv::Mat map1x(1, 1, CV_16U, cv::Scalar(0)), map1y(1, 1, CV_16U, cv::Scalar(0)), map2x(1, 1, CV_16U, cv::Scalar(0)),
      map2y(1, 1, CV_16U, cv::Scalar(0)), map3x(1, 1, CV_16U, cv::Scalar(0)), map3y(1, 1, CV_16U, cv::Scalar(0));
  SpatialTiff3 pos0{0.0f, 0.0f}, pos1{0.0f, 0.0f}, pos2{0.0f, 0.0f};
  std::vector<SpatialTiff3> positions = {pos0, pos1, pos2};

  ControlMasks3 masks;
  masks.img1_col = map1x;
  masks.img1_row = map1y;
  masks.img2_col = map2x;
  masks.img2_row = map2y;
  masks.img3_col = map3x;
  masks.img3_row = map3y;
  masks.whole_seam_mask_image = seam_mask_u;
  masks.positions = positions;
  ASSERT_TRUE(masks.is_valid());

  // Create three 1×1 float images (values arbitrary)
  using CudaMatF = hm::CudaMat<float>;
  cv::Mat host1(1, 1, CV_32F, cv::Scalar(5.0f));
  cv::Mat host2(1, 1, CV_32F, cv::Scalar(15.0f));
  cv::Mat host3(1, 1, CV_32F, cv::Scalar(25.0f));
  auto d_img1 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img2 = std::make_unique<CudaMatF>(1, 1, 1);
  auto d_img3 = std::make_unique<CudaMatF>(1, 1, 1);
  CUDA_CHECK(d_img1->upload(host1));
  CUDA_CHECK(d_img2->upload(host2));
  CUDA_CHECK(d_img3->upload(host3));

  auto d_canvas = std::make_unique<CudaMatF>(1, 1, 1);

  // Instantiate with num_levels=1 (soft‐seam) but mask is CV_8U → should fail in process().
  hm::pano::cuda::CudaStitchPano3<float, float> stitch(
      /*batch_size=*/1,
      /*num_levels=*/1,
      masks,
      /*match_exposure=*/false,
      /*quiet=*/true);
  ASSERT_TRUE(stitch.status().ok());

  auto resultOr = stitch.process(*d_img1, *d_img2, *d_img3, /*stream=*/0, std::move(d_canvas));
  ASSERT_FALSE(resultOr.ok());
}
#endif

// ----------------------------------------------------------------------------
// Main runner for Google Test
// ----------------------------------------------------------------------------
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
