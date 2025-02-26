#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include "FrameStabilizer.h"

namespace cupano {

FrameStabilizer::FrameStabilizer(const StabilizationParams& params)
    : mParams(params), mWidth(0), mHeight(0), mInitialized(false) {}

FrameStabilizer::~FrameStabilizer() {
  // All OpenCV smart pointers and GpuMat objects are automatically released.
}

bool FrameStabilizer::initialize(int width, int height) {
  mWidth = width;
  mHeight = height;

  // Create the GPU-based ORB detector (handles both keypoint detection and descriptor extraction)
  mORB = cv::cuda::ORB::create(mParams.maxFeatures);

  // Create a GPU-based descriptor matcher using brute-force matching with Hamming norm
  mMatcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);

  mInitialized = true;
  return true;
}

void FrameStabilizer::reset() {
  mTransformHistory.clear();
  mPrevKeypoints.clear();
  mPrevFrame.release();
  mPrevDescriptors.release();
}

bool FrameStabilizer::detectFeatures(
    const cv::cuda::GpuMat& frame,
    std::vector<cv::KeyPoint>& keypoints,
    cv::cuda::GpuMat& descriptors) {
  // Detect keypoints and compute descriptors on the GPU using ORB.
  mORB->detectAndCompute(frame, cv::noArray(), keypoints, descriptors);
  return !keypoints.empty();
}

bool FrameStabilizer::matchFeatures(
    const cv::cuda::GpuMat& prevDescriptors,
    const cv::cuda::GpuMat& currDescriptors,
    std::vector<cv::DMatch>& matches) {
  // Match descriptors between frames using the GPU matcher.
  mMatcher->match(prevDescriptors, currDescriptors, matches);

  // Optionally filter matches based on a distance threshold.
  const float maxDistance = 60.0f; // Adjust this threshold as needed.
  std::vector<cv::DMatch> goodMatches;
  for (const auto& match : matches) {
    if (match.distance < maxDistance)
      goodMatches.push_back(match);
  }
  matches.swap(goodMatches);
  return !matches.empty();
}

bool FrameStabilizer::estimateTransformation(
    const std::vector<cv::KeyPoint>& prevKeypoints,
    const std::vector<cv::KeyPoint>& currKeypoints,
    const std::vector<cv::DMatch>& matches,
    cv::Mat& transformMatrix) {
  if (matches.size() < 4) {
    transformMatrix = cv::Mat::eye(3, 3, CV_64F);
    return false;
  }

  // Prepare point correspondences.
  std::vector<cv::Point2f> prevPoints, currPoints;
  for (const auto& match : matches) {
    prevPoints.push_back(prevKeypoints[match.queryIdx].pt);
    currPoints.push_back(currKeypoints[match.trainIdx].pt);
  }

  // Compute the homography using RANSAC.
  std::vector<uchar> inliersMask;
  transformMatrix = cv::findHomography(currPoints, prevPoints, cv::RANSAC, 3.0, inliersMask);

  // Count inliers to verify reliability.
  int inliers = 0;
  for (auto mask : inliersMask) {
    if (mask)
      inliers++;
  }
  if (inliers < 4) {
    transformMatrix = cv::Mat::eye(3, 3, CV_64F);
    return false;
  }

  return true;
}

bool FrameStabilizer::stabilizeFrame(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
  if (!mInitialized || inputFrame.empty())
    return false;

  // Convert the frame to grayscale if needed.
  cv::cuda::GpuMat grayFrame;
  if (inputFrame.channels() == 3) {
    cv::cuda::cvtColor(inputFrame, grayFrame, cv::COLOR_BGR2GRAY);
  } else {
    grayFrame = inputFrame;
  }

  // Detect features and compute descriptors using ORB.
  std::vector<cv::KeyPoint> currKeypoints;
  cv::cuda::GpuMat currDescriptors;
  if (!detectFeatures(grayFrame, currKeypoints, currDescriptors))
    return false;

  // For the first frame, store the features and return the original frame.
  if (mPrevFrame.empty() || mPrevKeypoints.empty() || mPrevDescriptors.empty()) {
    grayFrame.copyTo(mPrevFrame);
    mPrevKeypoints = currKeypoints;
    currDescriptors.copyTo(mPrevDescriptors);
    inputFrame.copyTo(outputFrame);
    return true;
  }

  // Match features between the previous and current frames.
  std::vector<cv::DMatch> matches;
  if (!matchFeatures(mPrevDescriptors, currDescriptors, matches) || matches.size() < 4) {
    // If not enough matches, update previous data and return the original frame.
    grayFrame.copyTo(mPrevFrame);
    mPrevKeypoints = currKeypoints;
    currDescriptors.copyTo(mPrevDescriptors);
    inputFrame.copyTo(outputFrame);
    return true;
  }

  // Estimate the transformation from matched features.
  cv::Mat transformMatrix;
  if (!estimateTransformation(mPrevKeypoints, currKeypoints, matches, transformMatrix)) {
    inputFrame.copyTo(outputFrame);
    return true;
  }

  // Update the transformation history.
  mTransformHistory.push_back(transformMatrix);
  if (mTransformHistory.size() > mParams.historyLength) {
    mTransformHistory.erase(mTransformHistory.begin());
  }

  // Compute a smoothed transformation by combining history.
  cv::Mat smoothTransform = cv::Mat::eye(3, 3, CV_64F);
  for (const auto& t : mTransformHistory) {
    smoothTransform = smoothTransform.mul(t);
  }
  if (!mTransformHistory.empty()) {
    double scale = 1.0 / mTransformHistory.size();
    smoothTransform = smoothTransform.mul(scale);
  }

  // Warp the input frame using the smoothed transformation.
  cv::cuda::warpPerspective(inputFrame, outputFrame, smoothTransform, cv::Size(mWidth, mHeight), cv::INTER_LINEAR);

  // Update previous frame data.
  grayFrame.copyTo(mPrevFrame);
  mPrevKeypoints = currKeypoints;
  currDescriptors.copyTo(mPrevDescriptors);

  return true;
}

} // namespace cupano
