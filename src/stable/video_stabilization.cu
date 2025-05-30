// video_stabilization_streaming_median.cu
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <iostream>
#include <vector>

// CUDA kernel to perform a simple image warp based on a translation offset.
// Each thread maps one pixel of the destination image.
__global__ void warpKernel(
    const unsigned char* input,
    unsigned char* output,
    int width,
    int height,
    int channels,
    float offsetX,
    float offsetY) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    int outIdx = (y * width + x) * channels;
    float srcX = x + offsetX;
    float srcY = y + offsetY;
    int srcXInt = int(roundf(srcX));
    int srcYInt = int(roundf(srcY));

    if (srcXInt >= 0 && srcXInt < width && srcYInt >= 0 && srcYInt < height) {
      int srcIdx = (srcYInt * width + srcXInt) * channels;
      for (int c = 0; c < channels; c++) {
        output[outIdx + c] = input[srcIdx + c];
      }
    } else {
      for (int c = 0; c < channels; c++) {
        output[outIdx + c] = 0;
      }
    }
  }
}

// Helper function that wraps the CUDA kernel launch.
void warpFrameCUDA(const cv::Mat& frame, cv::Mat& warpedFrame, float offsetX, float offsetY) {
  int width = frame.cols;
  int height = frame.rows;
  int channels = frame.channels();
  size_t numPixels = width * height;
  size_t imageSize = numPixels * channels * sizeof(unsigned char);

  unsigned char *d_input = nullptr, *d_output = nullptr;
  cudaMalloc(&d_input, imageSize);
  cudaMalloc(&d_output, imageSize);
  cudaMemcpy(d_input, frame.data, imageSize, cudaMemcpyHostToDevice);

  dim3 block(16, 16);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

  warpKernel<<<grid, block>>>(d_input, d_output, width, height, channels, offsetX, offsetY);
  cudaDeviceSynchronize();

  cudaMemcpy(warpedFrame.data, d_output, imageSize, cudaMemcpyDeviceToHost);
  cudaFree(d_input);
  cudaFree(d_output);
}

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: video_stabilization_streaming_median input_video output_video" << std::endl;
    return -1;
  }

  std::string inputVideo = argv[1];
  std::string outputVideo = argv[2];

  cv::VideoCapture cap(inputVideo);
  if (!cap.isOpened()) {
    std::cerr << "Error opening video file: " << inputVideo << std::endl;
    return -1;
  }

  int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = cap.get(cv::CAP_PROP_FPS);
  int fourcc = static_cast<int>(cap.get(cv::CAP_PROP_FOURCC));
  int totalFrames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

  // First pass: compute frame-to-frame transforms.
  std::vector<cv::Point2f> transforms;
  transforms.push_back(cv::Point2f(0, 0)); // First frame has no shift.

  cv::Mat prevFrame, prevGray;
  if (!cap.read(prevFrame)) {
    std::cerr << "Error reading first frame." << std::endl;
    return -1;
  }
  cv::cvtColor(prevFrame, prevGray, cv::COLOR_BGR2GRAY);
  prevGray.convertTo(prevGray, CV_32F);

  cv::Mat currFrame, currGray;
  int frameIdx = 1;
  while (cap.read(currFrame)) {
    cv::cvtColor(currFrame, currGray, cv::COLOR_BGR2GRAY);
    currGray.convertTo(currGray, CV_32F);

    cv::Point2d shift = cv::phaseCorrelate(prevGray, currGray);
    transforms.push_back(cv::Point2f(static_cast<float>(shift.x), static_cast<float>(shift.y)));

    prevGray = currGray.clone();
    std::cout << "Computed transform for frame " << frameIdx << " / " << totalFrames << "\r";
    frameIdx++;
  }
  std::cout << "\nFirst pass complete. Total transforms computed: " << transforms.size() << std::endl;

  // Compute cumulative camera trajectory.
  std::vector<cv::Point2f> trajectory(transforms.size());
  cv::Point2f cumulative(0, 0);
  for (size_t i = 0; i < transforms.size(); i++) {
    cumulative += transforms[i];
    trajectory[i] = cumulative;
  }

  // Smooth the trajectory using a median filter.
  std::vector<cv::Point2f> smoothedTrajectory(trajectory.size());
  int radius = 15; // Window radius for median filtering.
  for (size_t i = 0; i < trajectory.size(); i++) {
    std::vector<float> windowX, windowY;
    for (int j = -radius; j <= radius; j++) {
      int idx = static_cast<int>(i) + j;
      if (idx >= 0 && idx < static_cast<int>(trajectory.size())) {
        windowX.push_back(trajectory[idx].x);
        windowY.push_back(trajectory[idx].y);
      }
    }
    std::sort(windowX.begin(), windowX.end());
    std::sort(windowY.begin(), windowY.end());
    float medianX = windowX[windowX.size() / 2];
    float medianY = windowY[windowY.size() / 2];
    smoothedTrajectory[i] = cv::Point2f(medianX, medianY);
  }

  // Compute the correction needed for each frame.
  std::vector<cv::Point2f> corrections(trajectory.size());
  for (size_t i = 0; i < trajectory.size(); i++) {
    corrections[i] = smoothedTrajectory[i] - trajectory[i];
  }

  // Reset video capture for second pass.
  cap.release();
  cap.open(inputVideo);
  if (!cap.isOpened()) {
    std::cerr << "Error reopening video file: " << inputVideo << std::endl;
    return -1;
  }

  cv::VideoWriter writer;
  writer.open(outputVideo, fourcc, fps, cv::Size(frameWidth, frameHeight));
  if (!writer.isOpened()) {
    std::cerr << "Error opening video writer" << std::endl;
    return -1;
  }

  frameIdx = 0;
  while (cap.read(currFrame)) {
    cv::Mat stabilizedFrame(currFrame.size(), currFrame.type());
    float offsetX = 0, offsetY = 0;
    if (frameIdx < corrections.size()) {
      offsetX = corrections[frameIdx].x;
      offsetY = corrections[frameIdx].y;
    }

    warpFrameCUDA(currFrame, stabilizedFrame, offsetX, offsetY);
    writer.write(stabilizedFrame);

    std::cout << "Processed frame " << frameIdx + 1 << " / " << totalFrames << "\r";
    frameIdx++;
  }
  std::cout << "\nVideo stabilization complete. Output saved to " << outputVideo << std::endl;

  cap.release();
  writer.release();
  return 0;
}
