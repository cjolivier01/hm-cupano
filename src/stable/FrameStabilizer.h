#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#ifdef HAVE_OPENCV_CUDAARITHM
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>

namespace cupano {

struct KeyPoint {
    float x;
    float y;
    float response;
    float size;
    float angle;
};

struct StabilizationParams {
    int blockSize = 2;                // Harris corner block size
    int kSize = 3;                    // Sobel kernel size
    double k = 0.04;                  // Harris detector free parameter
    float qualityLevel = 0.01f;       // Minimum quality level for corners
    int maxCorners = 1000;            // Maximum number of corners to detect
    float minDistance = 10.0f;        // Minimum distance between corners
    int historyLength = 5;            // Number of frames to use for smoothing
    bool useSobel = true;             // Whether to use Sobel for gradient computation
};

class FrameStabilizer {
public:
    FrameStabilizer(const StabilizationParams& params = StabilizationParams());
    ~FrameStabilizer();

    // Initialize resources
    bool initialize(int width, int height);
    
    // Process a new frame and return stabilized version
    bool stabilizeFrame(const cv::cuda::GpuMat& inputFrame, cv::cuda::GpuMat& outputFrame);
    
    // Reset stabilization (use when scene changes)
    void reset();

private:
    // Detect corners using Harris detector
    bool detectHarrisCorners(const cv::cuda::GpuMat& frame, std::vector<KeyPoint>& keypoints);
    
    // Match features between frames
    bool matchFeatures(const std::vector<KeyPoint>& prevKeypoints, 
                       const std::vector<KeyPoint>& currKeypoints,
                       std::vector<cv::DMatch>& matches);
    
    // Estimate transformation matrix from matched features
    bool estimateTransformation(const std::vector<KeyPoint>& prevKeypoints,
                               const std::vector<KeyPoint>& currKeypoints,
                               const std::vector<cv::DMatch>& matches,
                               cv::Mat& transformMatrix);
    
    // Apply Sobel operator to compute gradients
    bool computeSobelGradients(const cv::cuda::GpuMat& frame, 
                              cv::cuda::GpuMat& gradX, 
                              cv::cuda::GpuMat& gradY);

    // CUDA kernels and device functions
    void runHarrisDetectorKernel(const cv::cuda::GpuMat& frame);
    void runSobelKernel(const cv::cuda::GpuMat& frame);

private:
    StabilizationParams mParams;
    int mWidth;
    int mHeight;
    
    // CUDA resources
    float* d_cornerResponses;
    float* d_tmpBuffer;
    
    // OpenCV objects for feature detection and matching
    cv::Ptr<cv::cuda::CornersDetector> mHarrisDetector;
    cv::Ptr<cv::DescriptorMatcher> mMatcher;
    
    // Transformation history for smoothing
    std::vector<cv::Mat> mTransformHistory;
    
    // Previous frame data
    cv::cuda::GpuMat mPrevFrame;
    std::vector<KeyPoint> mPrevKeypoints;
    
    // Initialization flag
    bool mInitialized;
};
#endif 

} // namespace cupano
