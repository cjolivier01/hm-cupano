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

TEST(CudaMatBasic, ExplicitStreamsAndHostCopies) {
  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  for (cudaStream_t selected : {cudaStream_t{}, stream}) {
    cv::Mat cpu(8, 16, CV_32FC1, cv::Scalar(17));
    CudaMat<float> uploaded(cpu, true, selected);
    EXPECT_EQ(cv::norm(cpu, uploaded.download(), cv::NORM_INF), 0);
    cpu.setTo(cv::Scalar(23));
    ASSERT_EQ(cudaSuccess, uploaded.upload(cpu));
    EXPECT_EQ(cv::norm(cpu, uploaded.download(), cv::NORM_INF), 0);
    CudaMat<float> batch(std::vector<cv::Mat>{cpu, cpu}, true, selected);
    EXPECT_EQ(cv::norm(cpu, batch.download(1), cv::NORM_INF), 0);

    CudaMat<float> gpu(1, cpu.cols, cpu.rows, 1, CUDA_PIXEL_FLOAT1, selected);
    ASSERT_EQ(
        cudaSuccess, cudaMemcpyAsync(gpu.data(), uploaded.data(), gpu.size(), cudaMemcpyDeviceToDevice, selected));
    CudaMat<float> view(gpu.data(), 1, cpu.cols, cpu.rows, 1, selected);
    EXPECT_FALSE(view.owns_memory());
    EXPECT_EQ(cv::norm(cpu, view.download(), cv::NORM_INF), 0);
  }
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());
}

#if GPU_BACKEND_CUDA
TEST(CudaMatBasic, ExplicitStreamAllocationAndFreeCanBeCaptured) {
  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  ASSERT_EQ(cudaSuccess, cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
  {
    CudaMat<float> gpu(1, 16, 8, 1, stream);
    ASSERT_TRUE(gpu.is_valid());
    ASSERT_EQ(cudaSuccess, cudaMemsetAsync(gpu.data(), 0, gpu.size(), stream));
  }
  cudaGraph_t graph;
  ASSERT_EQ(cudaSuccess, cudaStreamEndCapture(stream, &graph));
  cudaGraphExec_t executable;
  ASSERT_EQ(cudaSuccess, cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  ASSERT_EQ(cudaSuccess, cudaGraphLaunch(executable, stream));
  EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
  EXPECT_EQ(cudaSuccess, cudaGraphExecDestroy(executable));
  EXPECT_EQ(cudaSuccess, cudaGraphDestroy(graph));
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}
#endif
