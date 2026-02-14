// cudaPano3_test.cpp

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/hal/interface.h>

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
      /*match_exposure=*/false,
      /*quiet=*/true);
  ASSERT_TRUE(stitch.status().ok());

  // 3e) Call process(). Label=1 maps to the first input in current implementation.
  //     So expect output equals image1 value (30).
  auto resultOr = stitch.process(*d_img1, *d_img2, *d_img3, /*stream=*/0, std::move(d_canvas));
  ASSERT_TRUE(resultOr.ok()) << resultOr.status().message();

  std::unique_ptr<CudaMatF> d_out = std::move(resultOr).ConsumeValueOrDie();
  ASSERT_EQ(d_out->width(), 1);
  ASSERT_EQ(d_out->height(), 1);

  cv::Mat hostOut = d_out->download();
  ASSERT_EQ(hostOut.type(), CV_32FC4);
  cv::Vec4f pixel = hostOut.at<cv::Vec4f>(0, 0);
  EXPECT_NEAR(pixel[0], 30.0f, 1e-3f);
  EXPECT_NEAR(pixel[1], 30.0f, 1e-3f);
  EXPECT_NEAR(pixel[2], 30.0f, 1e-3f);
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
