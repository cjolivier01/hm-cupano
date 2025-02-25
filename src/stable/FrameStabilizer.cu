#include "FrameStabilizer.h"
#ifdef HAVE_OPENCV_CUDAARITHM
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

namespace cupano {

// CUDA kernels
namespace {
// CUDA kernel for Sobel edge detection
__global__ void sobelKernel(const float* src, float* dX, float* dY, int width, int height, int stride) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  // Calculate indices
  const int idx = y * stride + x;

  // Compute horizontal gradient (Sobel X)
  dX[idx] = -1.0f * src[(y - 1) * stride + (x - 1)] + -2.0f * src[y * stride + (x - 1)] +
      -1.0f * src[(y + 1) * stride + (x - 1)] + 1.0f * src[(y - 1) * stride + (x + 1)] +
      2.0f * src[y * stride + (x + 1)] + 1.0f * src[(y + 1) * stride + (x + 1)];

  // Compute vertical gradient (Sobel Y)
  dY[idx] = -1.0f * src[(y - 1) * stride + (x - 1)] + -2.0f * src[(y - 1) * stride + x] +
      -1.0f * src[(y - 1) * stride + (x + 1)] + 1.0f * src[(y + 1) * stride + (x - 1)] +
      2.0f * src[(y + 1) * stride + x] + 1.0f * src[(y + 1) * stride + (x + 1)];
}

// CUDA kernel for computing Harris corner response
__global__ void harrisResponseKernel(
    const float* dX,
    const float* dY,
    float* response,
    int width,
    int height,
    int stride,
    int blockSize,
    float k) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < blockSize || x >= width - blockSize || y < blockSize || y >= height - blockSize)
    return;

  const int idx = y * stride + x;

  // Compute components of the structure tensor
  float Ix2_sum = 0.0f, Iy2_sum = 0.0f, IxIy_sum = 0.0f;

  // Sum over the block
  for (int by = -blockSize; by <= blockSize; by++) {
    for (int bx = -blockSize; bx <= blockSize; bx++) {
      const int bidx = (y + by) * stride + (x + bx);
      const float Ix = dX[bidx];
      const float Iy = dY[bidx];

      Ix2_sum += Ix * Ix;
      Iy2_sum += Iy * Iy;
      IxIy_sum += Ix * Iy;
    }
  }

  // Harris response: det(M) - k * trace(M)^2
  const float det = Ix2_sum * Iy2_sum - IxIy_sum * IxIy_sum;
  const float trace = Ix2_sum + Iy2_sum;

  response[idx] = det - k * trace * trace;
}

// CUDA kernel for non-maximum suppression
__global__ void nonMaxSuppressionKernel(
    const float* response,
    float* maxima,
    int width,
    int height,
    int stride,
    float threshold) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= width - 1 || y < 1 || y >= height - 1)
    return;

  const int idx = y * stride + x;
  const float centerValue = response[idx];

  // Check if the center value is above threshold and is a local maximum
  if (centerValue <= threshold) {
    maxima[idx] = 0.0f;
    return;
  }

  // Check 8-neighborhood
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      if (dx == 0 && dy == 0)
        continue;

      const int nidx = (y + dy) * stride + (x + dx);
      if (response[nidx] > centerValue) {
        maxima[idx] = 0.0f;
        return;
      }
    }
  }

  maxima[idx] = centerValue;
}
} // namespace

// Constructor
FrameStabilizer::FrameStabilizer(const StabilizationParams& params)
    : mParams(params), mWidth(0), mHeight(0), d_cornerResponses(nullptr), d_tmpBuffer(nullptr), mInitialized(false) {}

// Destructor
FrameStabilizer::~FrameStabilizer() {
  if (d_cornerResponses) {
    cudaFree(d_cornerResponses);
  }
  if (d_tmpBuffer) {
    cudaFree(d_tmpBuffer);
  }
}

// Initialize resources
bool FrameStabilizer::initialize(int width, int height) {
  mWidth = width;
  mHeight = height;

  // Allocate device memory for corner responses
  cudaMalloc(&d_cornerResponses, width * height * sizeof(float));
  cudaMalloc(&d_tmpBuffer, width * height * sizeof(float) * 2); // For gradients

  // Create Harris detector and matcher
  mHarrisDetector = cv::cuda::createGoodFeaturesToTrackDetector(
      CV_8UC1, mParams.maxCorners, mParams.qualityLevel, mParams.minDistance);

  mMatcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

  mInitialized = true;
  return true;
}

// Reset stabilization
void FrameStabilizer::reset() {
  mTransformHistory.clear();
  mPrevKeypoints.clear();
  mPrevFrame.release();
}

// Process a new frame and return stabilized version
bool FrameStabilizer::stabilizeFrame(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame) {
  if (!mInitialized || inputFrame.empty()) {
    return false;
  }

  // Convert to grayscale if needed
  cv::cuda::GpuMat grayFrame;
  if (inputFrame.channels() == 3) {
    cv::cuda::cvtColor(inputFrame, grayFrame, cv::COLOR_BGR2GRAY);
  } else {
    grayFrame = inputFrame;
  }

  // Detect corners in current frame
  std::vector<KeyPoint> currKeypoints;
  if (!detectHarrisCorners(grayFrame, currKeypoints)) {
    return false;
  }

  // If this is the first frame, store it and return
  if (mPrevFrame.empty() || mPrevKeypoints.empty()) {
    grayFrame.copyTo(mPrevFrame);
    mPrevKeypoints = currKeypoints;
    inputFrame.copyTo(outputFrame);
    return true;
  }

  // Match features between frames
  std::vector<cv::DMatch> matches;
  if (!matchFeatures(mPrevKeypoints, currKeypoints, matches) || matches.size() < 4) {
    // Not enough matches, just copy the input
    inputFrame.copyTo(outputFrame);
    grayFrame.copyTo(mPrevFrame);
    mPrevKeypoints = currKeypoints;
    return true;
  }

  // Estimate transformation
  cv::Mat transformMatrix;
  if (!estimateTransformation(mPrevKeypoints, currKeypoints, matches, transformMatrix)) {
    inputFrame.copyTo(outputFrame);
    return true;
  }

  // Add to transformation history
  mTransformHistory.push_back(transformMatrix);
  if (mTransformHistory.size() > mParams.historyLength) {
    mTransformHistory.erase(mTransformHistory.begin());
  }

  // Compute smoothed transformation
  cv::Mat smoothTransform = cv::Mat::eye(3, 3, CV_64F);
  for (const auto& transform : mTransformHistory) {
    smoothTransform = smoothTransform.mul(transform);
  }
  if (mTransformHistory.size() > 0) {
    // Average the transformation
    double scale = 1.0 / mTransformHistory.size();
    smoothTransform = smoothTransform.mul(scale);
  }

  // Apply stabilization transform
  cv::cuda::warpPerspective(inputFrame, outputFrame, smoothTransform, cv::Size(mWidth, mHeight), cv::INTER_LINEAR);

  // Update previous frame data
  grayFrame.copyTo(mPrevFrame);
  mPrevKeypoints = currKeypoints;

  return true;
}

// Detect corners using Harris detector
bool FrameStabilizer::detectHarrisCorners(const cv::cuda::GpuMat& frame, std::vector<KeyPoint>& keypoints) {
  keypoints.clear();

  if (mParams.useSobel) {
    // Compute gradients using custom Sobel
    cv::cuda::GpuMat gradX(frame.size(), CV_32FC1);
    cv::cuda::GpuMat gradY(frame.size(), CV_32FC1);

    if (!computeSobelGradients(frame, gradX, gradY)) {
      return false;
    }

    // Compute Harris responses
    cv::cuda::GpuMat harrisResponses(frame.size(), CV_32FC1);

    // Run Harris response kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((frame.cols + blockSize.x - 1) / blockSize.x, (frame.rows + blockSize.y - 1) / blockSize.y);

    float* d_gradX = gradX.ptr<float>();
    float* d_gradY = gradY.ptr<float>();
    float* d_responses = harrisResponses.ptr<float>();

    harrisResponseKernel<<<gridSize, blockSize>>>(
        d_gradX, d_gradY, d_responses, frame.cols, frame.rows, frame.step1(), mParams.blockSize, mParams.k);

    // Non-maximum suppression
    cv::cuda::GpuMat cornerMaxima(frame.size(), CV_32FC1);
    float* d_maxima = cornerMaxima.ptr<float>();
    float threshold = 0.01f; // Adjust as needed

    nonMaxSuppressionKernel<<<gridSize, blockSize>>>(
        d_responses, d_maxima, frame.cols, frame.rows, frame.step1(), threshold);

    // Download maxima and create keypoints
    cv::Mat h_maxima;
    cornerMaxima.download(h_maxima);

    for (int y = 0; y < h_maxima.rows; y++) {
      for (int x = 0; x < h_maxima.cols; x++) {
        float response = h_maxima.at<float>(y, x);
        if (response > 0) {
          KeyPoint kp;
          kp.x = x;
          kp.y = y;
          kp.response = response;
          kp.size = 7.0f; // Default size
          keypoints.push_back(kp);

          // Limit number of keypoints
          if (keypoints.size() >= mParams.maxCorners) {
            break;
          }
        }
      }
      if (keypoints.size() >= mParams.maxCorners) {
        break;
      }
    }
  } else {
    // Use OpenCV's CUDA implementation
    cv::cuda::GpuMat d_corners;
    std::vector<cv::KeyPoint> cvKeypoints;

    // Detect corners and download them
    mHarrisDetector->detect(frame, d_corners);

    // Download corners from GPU to CPU
    // Since downloadKeypoints isn't available in all OpenCV versions,
    // we'll manually convert the corners
    cv::Mat h_corners;
    d_corners.download(h_corners);

    // Convert to KeyPoint format based on the format returned by GoodFeaturesToTrackDetector
    // This typically returns a single-channel matrix with x,y coordinates
    if (!h_corners.empty()) {
      for (int i = 0; i < h_corners.rows; i++) {
        float x = h_corners.at<float>(i, 0);
        float y = h_corners.at<float>(i, 1);

        KeyPoint kp;
        kp.x = x;
        kp.y = y;
        kp.response = 1.0f; // Default response
        kp.size = 7.0f; // Default size
        kp.angle = 0.0f; // Default angle
        keypoints.push_back(kp);
      }
    }
  }

  return !keypoints.empty();
}

// Match features between frames
bool FrameStabilizer::matchFeatures(
    const std::vector<KeyPoint>& prevKeypoints,
    const std::vector<KeyPoint>& currKeypoints,
    std::vector<cv::DMatch>& matches) {
  // Convert KeyPoints to cv::KeyPoint
  std::vector<cv::KeyPoint> cvPrevKeypoints, cvCurrKeypoints;

  for (const auto& kp : prevKeypoints) {
    cvPrevKeypoints.emplace_back(cv::KeyPoint(cv::Point2f(kp.x, kp.y), kp.size, kp.angle, kp.response));
  }

  for (const auto& kp : currKeypoints) {
    cvCurrKeypoints.emplace_back(cv::KeyPoint(cv::Point2f(kp.x, kp.y), kp.size, kp.angle, kp.response));
  }

  // For simplicity, we'll use a simple distance-based matching
  // In a real implementation, you'd compute descriptors and use proper matching

  // Simple matching based on Euclidean distance
  const float maxDistance = 30.0f; // Maximum distance for a valid match

  for (size_t i = 0; i < cvPrevKeypoints.size(); i++) {
    float minDist = std::numeric_limits<float>::max();
    int bestMatch = -1;

    for (size_t j = 0; j < cvCurrKeypoints.size(); j++) {
      float dx = cvPrevKeypoints[i].pt.x - cvCurrKeypoints[j].pt.x;
      float dy = cvPrevKeypoints[i].pt.y - cvCurrKeypoints[j].pt.y;
      float dist = std::sqrt(dx * dx + dy * dy);

      if (dist < minDist) {
        minDist = dist;
        bestMatch = j;
      }
    }

    if (bestMatch >= 0 && minDist < maxDistance) {
      matches.emplace_back(cv::DMatch(i, bestMatch, minDist));
    }
  }

  return !matches.empty();
}

// Estimate transformation matrix from matched features
bool FrameStabilizer::estimateTransformation(
    const std::vector<KeyPoint>& prevKeypoints,
    const std::vector<KeyPoint>& currKeypoints,
    const std::vector<cv::DMatch>& matches,
    cv::Mat& transformMatrix) {
  if (matches.size() < 4) {
    // Need at least 4 point pairs for homography
    transformMatrix = cv::Mat::eye(3, 3, CV_64F);
    return false;
  }

  // Prepare point pairs
  std::vector<cv::Point2f> prevPoints, currPoints;
  for (const auto& match : matches) {
    prevPoints.push_back(cv::Point2f(prevKeypoints[match.queryIdx].x, prevKeypoints[match.queryIdx].y));
    currPoints.push_back(cv::Point2f(currKeypoints[match.trainIdx].x, currKeypoints[match.trainIdx].y));
  }

  // Find homography with RANSAC
  std::vector<uchar> inliersMask;
  transformMatrix = cv::findHomography(currPoints, prevPoints, cv::RANSAC, 3.0, inliersMask);

  // Count inliers
  int inliers = 0;
  for (auto mask : inliersMask) {
    if (mask)
      inliers++;
  }

  // If not enough inliers, return identity
  if (inliers < 4) {
    transformMatrix = cv::Mat::eye(3, 3, CV_64F);
    return false;
  }

  return true;
}

// Apply Sobel operator to compute gradients
bool FrameStabilizer::computeSobelGradients(
    const cv::cuda::GpuMat& frame,
    cv::cuda::GpuMat& gradX,
    cv::cuda::GpuMat& gradY) {
  // Convert to float if needed
  cv::cuda::GpuMat floatFrame;
  if (frame.type() != CV_32FC1) {
    frame.convertTo(floatFrame, CV_32FC1, 1.0 / 255.0);
  } else {
    floatFrame = frame;
  }

  // Launch Sobel kernel
  dim3 blockSize(16, 16);
  dim3 gridSize((frame.cols + blockSize.x - 1) / blockSize.x, (frame.rows + blockSize.y - 1) / blockSize.y);

  float* d_src = floatFrame.ptr<float>();
  float* d_gradX = gradX.ptr<float>();
  float* d_gradY = gradY.ptr<float>();

  sobelKernel<<<gridSize, blockSize>>>(d_src, d_gradX, d_gradY, frame.cols, frame.rows, frame.step1());

  // Check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    return false;
  }

  return true;
}

} // namespace cupano
#endif // HAVE_OPENCV_CUDAARITHM
