#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/controlMasks.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano.h"
#include "cupano/utils/showImage.h"

#include <cuda_runtime.h> // for CUDA vector types
#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

#include <cuda_runtime.h>
// #if (CUDART_VERSION >= 11000)
// #include <cuda_bf16.h>
// #endif
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv2/core/hal/interface.h>
#include <opencv2/highgui.hpp>

#include <opencv2/imgcodecs.hpp>

#include <fcntl.h>
#include <getopt.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  std::vector<cv::Mat> batch;
  batch.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i) {
    batch.emplace_back(mat.clone());
  }
  return batch;
}

int main(int argc, char** argv) {
  // Usage check.

  bool perf = false;
  bool show = false;
  bool adjust_images = false;
  std::string game_id;
  std::string directory;
  std::string output;
  int device_id = 0;
  int batch_size = 1;

#ifdef __aarch64__
  int num_levels = 0;
#else
  int num_levels = 6;
#endif

  // Define the long options.
  // The 'val' field provides a short option equivalent.
  static struct option long_options[] = {
      {"show", no_argument, 0, 's'}, // --show (flag)
      {"perf", no_argument, 0, 'p'}, // --perf (flag)
      {"game-id", required_argument, 0, 'g'}, // --game-id <value>
      {"levels", required_argument, 0, 'l'}, // --levels <value>
      {"adjust", required_argument, 0, 'a'}, // --directory <value>
      {"directory", required_argument, 0, 'd'}, // --directory <value>
      {"output", required_argument, 0, 'o'}, // --output <value>
      {"cuda-device", required_argument, 0, 'c'}, // --cuda-device <value>
      {"batch-size", required_argument, 0, 'b'}, // --batch-size <value>
      {0, 0, 0, 0} // End-of-array marker
  };

  // The short options string:
  // 's' for --show (no argument),
  // 'g:' means option 'g' requires an argument,
  // 'd:' means option 'd' requires an argument.
  const char* short_opts = "spg:d:o:c:a:l:";

  int option_index = 0;
  int opt;
  // Loop through and parse each option.
  while ((opt = getopt_long(argc, argv, short_opts, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'l': // --levels
        num_levels = std::atoi(optarg);
        break;
      case 'b':
        batch_size = std::atoi(optarg);
        break;
      case 'a': // --adjust
        adjust_images = !!std::atoi(optarg);
        break;
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
            << " [--show] [--perf] [--game-id <id>] [--directory <dir>] [--cuda-device <value>] [--output <value>] [--adjust <0|1>] [--levels <value>]"
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
  if (sample_img_left.empty()) {
    std::cerr << "Unable to load image: " << sample_img_left_path << std::endl;
    return 1;
  }
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  if (sample_img_left.empty()) {
    std::cerr << "Unable to load image: " << sample_img_left_path << std::endl;
    return 1;
  }

  hm::pano::ControlMasks control_masks;
  control_masks.load(directory);

  // Configurable parameter: number of pyramid levels.
#if 1
#if 1
  // using T_pipeline = uchar4;
  using T_pipeline = uchar3;
  // using T_pipeline = float3;
  // using T_compute = float4;
  // using T_compute = float4;
  using T_compute = half4;
  // using T_compute = half3;
#else
  using T_pipeline = float3;
  using T_compute = float3;
#endif
#else
  using T_pipeline = float;
  using T_compute = __half;
#endif

  hm::pano::cuda::CudaStitchPano<T_pipeline, T_compute> pano(
      batch_size, num_levels, control_masks, /*match_exposure=*/adjust_images);

  std::cout << "Canvas size: " << pano.canvas_width() << " x " << pano.canvas_height() << std::endl;

  const int cvPipelineType = cudaPixelTypeToCvType(hm::CudaTypeToPixelType<T_pipeline>::value);

  if (sample_img_left.type() != cvPipelineType) {
    if (std::is_floating_point<BaseScalar_t<T_pipeline>>()) {
      sample_img_left.convertTo(sample_img_left, cvPipelineType, 1.0 / 255.0);
      sample_img_right.convertTo(sample_img_right, cvPipelineType, 1.0 / 255.0);
    } else {
      if (sizeof(T_pipeline) / sizeof(BaseScalar_t<T_pipeline>) == 4) {
        cv::cvtColor(sample_img_left, sample_img_left, cv::COLOR_BGR2BGRA);
        cv::cvtColor(sample_img_right, sample_img_right, cv::COLOR_BGR2BGRA);
      }
    }
  }

  // cv::imshow("", sample_img_right);
  // cv::waitKey(0);

  hm::CudaMat<T_pipeline> inputImage1(as_batch(sample_img_left, batch_size));
  hm::CudaMat<T_pipeline> inputImage2(as_batch(sample_img_right, batch_size));

  auto canvas = std::make_unique<hm::CudaMat<T_pipeline>>(pano.batch_size(), pano.canvas_width(), pano.canvas_height());

  auto blendedCanvasResult = pano.process(inputImage1, inputImage2, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  canvas = blendedCanvasResult.ConsumeValueOrDie();

  cudaStreamSynchronize(stream);

  if (!output.empty()) {
    cv::imwrite(output, canvas->download());
  }
  if (show) {
    SHOW_SCALED_BATCH_ITEM(canvas, 0.25, 1);
    // SHOW_SCALED(canvas, 1.0);
    // hm::utils::show_surface("Canvas", canvas->surface(), /*wait=*/true);
    usleep(10000);
  }

  if (perf) {
    auto start_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();

    size_t frame_count = 100;
    for (size_t i = 0; i < frame_count; ++i) {
      canvas = pano.process(inputImage1, inputImage2, stream, std::move(canvas)).ConsumeValueOrDie();
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
