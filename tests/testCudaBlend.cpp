#include "canvasManager.h"
#include "controlMasks.h"
#include "cudaBlend.h"
#include "cudaMat.h"
#include "cudaPano.h"
#include "cudaStatus.h"
#include "showImage.h"

#include <cuda_runtime.h> // for CUDA vector types
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/imgcodecs.hpp>

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  return std::vector<cv::Mat>(batch_size, mat);
}

int main(int argc, char** argv) {
  // Usage check.

  bool perf = false;
  bool show = false;
  std::string game_id;
  std::string directory;
  std::string output;
  int device_id = 0;

  // Define the long options.
  // The 'val' field provides a short option equivalent.
  static struct option long_options[] = {
      {"show", no_argument, 0, 's'}, // --show (flag)
      {"perf", no_argument, 0, 'p'}, // --perf (flag)
      {"game-id", required_argument, 0, 'g'}, // --game-id <value>
      {"directory", required_argument, 0, 'd'}, // --directory <value>
      {"output", required_argument, 0, 'o'}, // --output <value>
      {"cuda-device", required_argument, 0, 'c'}, // --cuda-device <value>
      {0, 0, 0, 0} // End-of-array marker
  };

  // The short options string:
  // 's' for --show (no argument),
  // 'g:' means option 'g' requires an argument,
  // 'd:' means option 'd' requires an argument.
  const char* short_opts = "spg:d:o:c:";

  int option_index = 0;
  int opt;
  // Loop through and parse each option.
  while ((opt = getopt_long(argc, argv, short_opts, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'p': // --perf
        perf = true;
        break;
      case 's': // --show
        show = true;
        break;
      case 'c': // --cuda-device
        device_id = std::atoi(optarg);
        break;
      case 'g': // --game-id
        game_id = optarg;
        break;
      case 'd': // --directory
        directory = optarg;
        break;
      case 'o': // --directory
        output = optarg;
        break;
      case '?': // Unknown option or missing required argument.
        std::cerr
            << "Usage: " << argv[0]
            << " [--show] [--perf] [--game-id <id>] [--directory <dir>] [--cuda-device <value>] [--output <value>]"
            << std::endl;
        exit(EXIT_FAILURE);
      default:
        break;
    }
  }

  // (Optional) Process any remaining non-option arguments.
  if (optind < argc) {
    std::cout << "Non-option arguments: ";
    for (int i = optind; i < argc; ++i)
      std::cout << argv[i] << " ";
    std::cout << std::endl;
  }

  cudaSetDevice(device_id);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  if (directory.empty()) {
    directory = std::string(::getenv("HOME")) + "/Videos";
  }
  if (!game_id.empty()) {
    directory += std::string("/") + game_id;
  }

  // Left video frame
  std::string sample_img_left_path = directory + "/left.png";
  // Right video frame
  std::string sample_img_right_path = directory + "/right.png";

  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  hm::pano::ControlMasks control_masks;
  control_masks.load(directory);

// Configurable parameter: number of pyramid levels.
#ifdef __aarch64__
  // Lower compute, quick and dirty
  int numLevels = 0;
  // int numLevels = 6;
#else
  // int numLevels = 6;
  // int numLevels = 2;
  // int numLevels = 6;
  int numLevels = 0;
#endif

#if 1
#if 1
  using T_pipeline = uchar3;
  // using T_pipeline = float3;

  using T_compute = float3;
  // using T_compute = half3;
#else
  using T_pipeline = float3;
  using T_compute = float3;
#endif
#else
  using T_pipeline = float;
  using T_compute = __half;
#endif

  constexpr int kBatchSize = 1;
  // constexpr int kBatchSize = 2;

  hm::pano::cuda::CudaStitchPano<T_pipeline, T_compute> pano(kBatchSize, numLevels, control_masks);

  std::cout << "Canvas size: " << pano.canvas_width() << " x " << pano.canvas_height() << std::endl;

  const int CV_T_PIPELINE = cudaPixelTypeToCvType(CudaTypeToPixelType<T_pipeline>::value);

  cv::Scalar offset = pano.match_seam_images(
                              sample_img_left,
                              sample_img_right,
                              control_masks.whole_seam_mask_image,
                              10,
                              cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
                              cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos))
                          .value();

  if (std::is_floating_point_v<BaseScalar_t<T_pipeline>>) {
    sample_img_left.convertTo(sample_img_left, CV_T_PIPELINE, 1.0 / 255.0);
    sample_img_right.convertTo(sample_img_right, CV_T_PIPELINE, 1.0 / 255.0);
  }

  CudaMat<T_pipeline> sampleImage1(as_batch(sample_img_left, kBatchSize));
  CudaMat<T_pipeline> sampleImage2(as_batch(sample_img_right, kBatchSize));

  auto canvas = std::make_unique<CudaMat<T_pipeline>>(pano.batch_size(), pano.canvas_width(), pano.canvas_height());

  auto blendedCanvasResult = pano.process(sampleImage1, sampleImage2, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  canvas = blendedCanvasResult.ConsumeValueOrDie();

  if (!output.empty()) {
    cv::imwrite(output, canvas->download());
  }
  if (show) {
    // SHOW_SMALL(canvas);
    // SHOW_IMAGE(canvas);
    SHOW_SCALED(canvas, 0.25);
    usleep(10000);
  }

  if (perf) {
    auto start_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();

    size_t frame_count = 100;
    for (size_t i = 0; i < frame_count; ++i) {
      canvas = pano.process(sampleImage1, sampleImage2, stream, std::move(canvas)).ConsumeValueOrDie();
      cudaStreamSynchronize(stream);
    }

    auto stop_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    float ms = stop_ms - start_ms;
    float sec_per_frame = (ms / 1000) / (frame_count * pano.batch_size());
    std::cout << "Blend speed: " << (1.0 / sec_per_frame) << "fps" << std::endl;
  }
  cudaStreamDestroy(stream);

  return cudaSuccess;
}
