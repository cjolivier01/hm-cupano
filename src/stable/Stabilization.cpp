// main.cpp (or wherever your video processing loop is)
#include "stable/FrameStabilizer.h"
// Example usage in your video processing pipeline:
bool processVideoWithStabilization(const std::string& inputPath, const std::string& outputPath) {
  // Open video
  cv::VideoCapture cap(inputPath);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open input video: " << inputPath << std::endl;
    return false;
  }

  // Get video properties
  int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  double fps = cap.get(cv::CAP_PROP_FPS);
  int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G'); // Or other codec

  // Create video writer
  cv::VideoWriter writer(outputPath, fourcc, fps, cv::Size(width, height));
  if (!writer.isOpened()) {
    std::cerr << "Failed to open output video: " << outputPath << std::endl;
    return false;
  }

  // Create frame stabilizer
  cupano::StabilizationParams params;
  params.blockSize = 2;
  params.kSize = 3;
  params.k = 0.04;
  params.qualityLevel = 0.01f;
  params.maxCorners = 1000;
  params.minDistance = 10.0f;
  params.historyLength = 5;
  params.useSobel = true;

  cupano::FrameStabilizer stabilizer(params);
  if (!stabilizer.initialize(width, height)) {
    std::cerr << "Failed to initialize stabilizer" << std::endl;
    return false;
  }

  // Process video frames
  cv::Mat frame, stabilizedFrame;
  cv::cuda::GpuMat d_frame, d_stabilizedFrame;

  while (cap.read(frame)) {
    // Upload to GPU
    d_frame.upload(frame);

    // Apply stabilization
    if (!stabilizer.stabilizeFrame(d_frame, d_stabilizedFrame)) {
      std::cerr << "Failed to stabilize frame" << std::endl;
      d_stabilizedFrame = d_frame; // Use original frame if stabilization fails
    }

    // Download result
    d_stabilizedFrame.download(stabilizedFrame);

    // Write to output video
    writer.write(stabilizedFrame);

    // Show progress (optional)
    std::cout << "Processed frame: " << cap.get(cv::CAP_PROP_POS_FRAMES) << "\r" << std::flush;
  }

  // Release resources
  cap.release();
  writer.release();

  std::cout << "Video stabilization complete: " << outputPath << std::endl;
  return true;
}

// Example command line integration
int main(int argc, char** argv) {
  if (argc < 3) {
    std::cout << "Usage: " << argv[0] << " <input_video> <output_video>" << std::endl;
    return -1;
  }

  std::string inputPath = argv[1];
  std::string outputPath = argv[2];

  if (!processVideoWithStabilization(inputPath, outputPath)) {
    std::cerr << "Failed to process video" << std::endl;
    return -1;
  }

  return 0;
}
