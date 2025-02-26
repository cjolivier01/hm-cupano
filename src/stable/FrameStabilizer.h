#pragma once

#include <cuda_runtime.h>

#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

namespace cupano {

// Parameters for stabilization using GPU‑based ORB
struct StabilizationParams {
  int maxFeatures = 1000;  // Maximum number of features to detect
  int historyLength = 5;   // Number of frames to use for smoothing
  int blockSize = 2;
  int kSize = 3;
  int k = 0.04;
  float qualityLevel = 0.01f;
  int maxCorners = 1000;
  float minDistance = 10.0f;
  bool useSobel = true;
};

class FrameStabilizer {
 public:
  FrameStabilizer(const StabilizationParams& params = StabilizationParams());
  ~FrameStabilizer();

  // Initialize resources with frame width and height
  bool initialize(int width, int height);

  // Process a new frame and output a stabilized version
  bool stabilizeFrame(const cv::cuda::GpuMat& inputFrame,
                      cv::cuda::GpuMat& outputFrame);

  // Reset stabilization history (e.g. when scene changes)
  void reset();

 private:
  // Detect features and compute descriptors using ORB on the GPU
  bool detectFeatures(const cv::cuda::GpuMat& frame,
                      std::vector<cv::KeyPoint>& keypoints,
                      cv::cuda::GpuMat& descriptors);

  // Match features between frames using a GPU-based descriptor matcher
  bool matchFeatures(const cv::cuda::GpuMat& prevDescriptors,
                     const cv::cuda::GpuMat& currDescriptors,
                     std::vector<cv::DMatch>& matches);

  // Estimate the homography transformation between matched features
  bool estimateTransformation(const std::vector<cv::KeyPoint>& prevKeypoints,
                              const std::vector<cv::KeyPoint>& currKeypoints,
                              const std::vector<cv::DMatch>& matches,
                              cv::Mat& transformMatrix);

 private:
  StabilizationParams mParams;
  int mWidth;
  int mHeight;

  // GPU-based ORB detector and descriptor matcher
  cv::Ptr<cv::cuda::ORB> mORB;
  cv::Ptr<cv::cuda::DescriptorMatcher> mMatcher;

  // History of transformation matrices for smoothing
  std::vector<cv::Mat> mTransformHistory;

  // Previous frame data: grayscale image, keypoints, and descriptors
  cv::cuda::GpuMat mPrevFrame;
  std::vector<cv::KeyPoint> mPrevKeypoints;
  cv::cuda::GpuMat mPrevDescriptors;

  // Initialization flag
  bool mInitialized;
};

}  // namespace cupano
