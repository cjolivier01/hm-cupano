#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/controlMasksN.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPanoN.h"
#include "cupano/utils/showImage.h"

#include "cupano/gpu/gpu_runtime.h"
#include <opencv2/opencv.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <type_traits>

#include <getopt.h>
#include <unistd.h>

static std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  std::vector<cv::Mat> batch;
  batch.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i)
    batch.emplace_back(mat.clone());
  return batch;
}

int main(int argc, char** argv) {
  bool perf = false;
  bool show = false;
  bool adjust_images = false;
  std::string game_id;
  std::string directory;
  std::string output;
  int device_id = 0;
  int batch_size = 1;
  int num_images = 3; // default to 3, supports N >= 2

#ifdef __aarch64__
  int num_levels = 0;
#else
  int num_levels = 6;
#endif

  static struct option long_options[] = {
      {"show", no_argument, 0, 's'},
      {"perf", no_argument, 0, 'p'},
      {"game-id", required_argument, 0, 'g'},
      {"levels", required_argument, 0, 'l'},
      {"adjust", required_argument, 0, 'a'},
      {"directory", required_argument, 0, 'd'},
      {"output", required_argument, 0, 'o'},
      {"cuda-device", required_argument, 0, 'c'},
      {"batch-size", required_argument, 0, 'b'},
      {"num-images", required_argument, 0, 'n'},
      {0, 0, 0, 0}};

  const char* short_opts = "spg:d:o:c:a:l:b:n:";
  int option_index = 0;
  int opt;
  while ((opt = getopt_long(argc, argv, short_opts, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'l':
        num_levels = std::atoi(optarg);
        break;
      case 'b':
        batch_size = std::atoi(optarg);
        break;
      case 'a':
        adjust_images = !!std::atoi(optarg);
        break;
      case 'p':
        perf = true;
        break;
      case 's':
        show = true;
        break;
      case 'c':
        device_id = std::atoi(optarg);
        break;
      case 'g':
        game_id = optarg;
        break;
      case 'd':
        directory = optarg;
        break;
      case 'o':
        output = optarg;
        break;
      case 'n':
        num_images = std::max(2, std::atoi(optarg));
        break;
      case '?':
        std::cerr
            << "Usage: " << argv[0]
            << " [--show] [--perf] [--game-id <id>] [--directory <dir>] [--cuda-device <v>] [--output <v>] [--adjust <0|1>] [--levels <v>] [--num-images <N>]"
            << std::endl;
        return 2;
      default:
        break;
    }
  }

  cudaSetDevice(device_id);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  if (directory.empty())
    directory = std::string(::getenv("HOME")) + "/Videos";
  if (!game_id.empty())
    directory += std::string("/") + game_id;

  // Load N input images: image0.png .. image{N-1}.png
  std::vector<cv::Mat> imgs_cv;
  imgs_cv.reserve(num_images);
  for (int i = 0; i < num_images; ++i) {
    std::string path = directory + "/image" + std::to_string(i) + ".png";
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
      std::cerr << "Unable to load image: " << path << std::endl;
      return 1;
    }
    imgs_cv.push_back(std::move(img));
  }

  hm::pano::ControlMasksN control_masks;
  if (!control_masks.load(directory, num_images)) {
    std::cerr << "Failed to load control masks from directory: " << directory << "; errno: " << strerror(errno)
              << std::endl;
    return 1;
  }

  using T_pipeline = uchar4;
  using T_compute = float4;

  hm::pano::cuda::CudaStitchPanoN<T_pipeline, T_compute> pano(
      batch_size, num_levels, control_masks, /*match_exposure=*/adjust_images);
  std::cout << "Canvas size: " << pano.canvas_width() << " x " << pano.canvas_height() << std::endl;

  const int cvPipelineType = cudaPixelTypeToCvType(hm::CudaTypeToPixelType<T_pipeline>::value);
  for (auto& img : imgs_cv) {
    if (img.type() != cvPipelineType) {
      if (std::is_floating_point<BaseScalar_t<T_pipeline>>()) {
        img.convertTo(img, cvPipelineType, 1.0 / 255.0);
      } else {
        if (sizeof(T_pipeline) / sizeof(BaseScalar_t<T_pipeline>) == 4) {
          cv::cvtColor(img, img, cv::COLOR_BGR2BGRA);
        }
      }
    }
  }

  // Upload to device and prepare pointers
  std::vector<std::unique_ptr<hm::CudaMat<T_pipeline>>> inputs;
  std::vector<const hm::CudaMat<T_pipeline>*> input_ptrs;
  inputs.reserve(num_images);
  input_ptrs.reserve(num_images);
  for (int i = 0; i < num_images; ++i) {
    inputs.emplace_back(std::make_unique<hm::CudaMat<T_pipeline>>(as_batch(imgs_cv[i], batch_size)));
    input_ptrs.push_back(inputs.back().get());
  }

  auto canvas = std::make_unique<hm::CudaMat<T_pipeline>>(pano.batch_size(), pano.canvas_width(), pano.canvas_height());

  auto blendedCanvasResult = pano.process(input_ptrs, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  canvas = blendedCanvasResult.ConsumeValueOrDie();

  cudaStreamSynchronize(stream);

  if (!output.empty())
    cv::imwrite(output, canvas->download());
  if (show) {
    SHOW_SCALED(canvas, 1);
    usleep(10000);
  }

  if (perf) {
    auto start_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
            .count();
    size_t frame_count = 100;
    for (size_t i = 0; i < frame_count; ++i) {
      canvas = pano.process(input_ptrs, stream, std::move(canvas)).ConsumeValueOrDie();
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
