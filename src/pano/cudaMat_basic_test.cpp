#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <cupano/pano/cudaMat.h>

using namespace hm;

TEST(CudaMatBasic, UploadDownloadUchar3) {
  // Create a small 2x2 BGR image
  cv::Mat cpu(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));

  // Construct GPU image from cv::Mat
  CudaMat<uchar3> gpu(cpu, /*copy=*/true);
  ASSERT_TRUE(gpu.is_valid());
  EXPECT_EQ(gpu.width(), 2);
  EXPECT_EQ(gpu.height(), 2);
  EXPECT_EQ(gpu.batch_size(), 1);

  // Round-trip back to CPU
  cv::Mat rt = gpu.download(0);
  EXPECT_EQ(rt.rows, 2);
  EXPECT_EQ(rt.cols, 2);
  EXPECT_EQ(rt.type(), cpu.type());
}

