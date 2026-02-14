/**
 * @file stitch_three_videos.cpp
 * @brief Sample app to stitch three input videos into a panorama using hm::cupano.
 *
 * Demonstrates a complete I/O loop around the 3-camera CUDA panorama stitcher.
 * Attempts GPU-accelerated I/O via cudacodec when requested and available,
 * falling back to CPU I/O otherwise. Optionally previews the stitched result.
 *
 * Key options:
 *  - --left/--middle/--right (or --video0/1/2): input video paths
 *  - --control: directory with control mask/remap files
 *  - --output: output video path
 *  - --gpu-decode/--gpu-encode: prefer GPU reader/writer (0|1)
 *  - --levels: number of pyramid levels for the stitcher
 *  - --adjust: exposure matching (0|1)
 *  - --cuda-device: CUDA device id
 *  - --max-frames: limit processed frames (smoke tests)
 *  - --show / --show-scaled: optional display
 */

#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/controlMasks3.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/utils/showImage.h"

#include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#if defined(__has_include)
#if __has_include(<opencv2/cudacodec.hpp>) && __has_include(<opencv2/cudaimgproc.hpp>)
#define HM_HAS_OPENCV_CUDA_IO 1
#else
#define HM_HAS_OPENCV_CUDA_IO 0
#endif
#else
#define HM_HAS_OPENCV_CUDA_IO 0
#endif

#if HM_HAS_OPENCV_CUDA_IO
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#endif
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <getopt.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

/**
 * @brief Abstraction over a video input source (CPU or GPU path).
 */
struct VideoInput {
  bool useCuda = false;                                        ///< True if using GPU reader.
  cv::VideoCapture cap;                                        ///< CPU reader.
#if HM_HAS_OPENCV_CUDA_IO
  cv::Ptr<cv::cudacodec::VideoReader> reader;                  ///< GPU reader.
#endif
  double fps = 0.0;                                            ///< Frames per second (best-effort).
  cv::Size size;                                               ///< Frame size, may be empty until first frame.
};

/**
 * @brief Read the next frame and convert to host BGRA.
 */
bool readNextBGRA(VideoInput& in, cv::Mat& outBGRA, cv::cuda::Stream& s) {
#if HM_HAS_OPENCV_CUDA_IO
  if (in.useCuda && in.reader) {
    cv::cuda::GpuMat d_bgr;
    if (!in.reader->nextFrame(d_bgr))
      return false;
    if (d_bgr.empty())
      return false;
    cv::cuda::GpuMat d_bgra;
    if (d_bgr.channels() == 3) {
      cv::cuda::cvtColor(d_bgr, d_bgra, cv::COLOR_BGR2BGRA, 0, s);
    } else if (d_bgr.channels() == 4) {
      d_bgra = d_bgr;
    } else if (d_bgr.channels() == 1) {
      cv::cuda::cvtColor(d_bgr, d_bgra, cv::COLOR_GRAY2BGRA, 0, s);
    } else {
      return false;
    }
    d_bgra.download(outBGRA, s);
    s.waitForCompletion();
    if (in.size.empty())
      in.size = d_bgra.size();
    return true;
  }
#endif

  cv::Mat bgr;
  if (!in.cap.read(bgr))
    return false;
  if (bgr.empty())
    return false;
  if (bgr.channels() == 3) {
    cv::cvtColor(bgr, outBGRA, cv::COLOR_BGR2BGRA);
  } else if (bgr.channels() == 4) {
    outBGRA = bgr;
  } else if (bgr.channels() == 1) {
    cv::cvtColor(bgr, outBGRA, cv::COLOR_GRAY2BGRA);
  } else {
    return false;
  }
  if (in.size.empty())
    in.size = outBGRA.size();
  return true;
}

/**
 * @brief Open a video input and prefer the GPU reader when requested.
 */
VideoInput openVideoInput(const std::string& path, bool preferCuda) {
  VideoInput vi;
  vi.size = {};
  vi.fps = 0.0;
#if HM_HAS_OPENCV_CUDA_IO
  if (preferCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    try {
      vi.reader = cv::cudacodec::createVideoReader(path);
      if (vi.reader) {
        vi.useCuda = true;
        double f = 0.0;
        if (vi.reader->get(cv::CAP_PROP_FPS, f))
          vi.fps = f;
        return vi;
      }
    } catch (const cv::Exception& e) {
      std::cerr << "Falling back to CPU reader for: " << path << " due to: " << e.what() << std::endl;
    }
  }
#endif
  if (!vi.cap.open(path)) {
    throw std::runtime_error("Failed to open video: " + path);
  }
  vi.useCuda = false;
  vi.fps = vi.cap.get(cv::CAP_PROP_FPS);
  vi.size = cv::Size((int)vi.cap.get(cv::CAP_PROP_FRAME_WIDTH), (int)vi.cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  return vi;
}

/**
 * @brief Abstraction over the output encoder (CPU or GPU path).
 */
struct VideoOutput {
  bool useCuda = false;                                       ///< True if using GPU writer.
  cv::VideoWriter cpu;                                        ///< CPU writer.
#if HM_HAS_OPENCV_CUDA_IO
  cv::Ptr<cv::cudacodec::VideoWriter> gpu;                    ///< GPU writer.
#endif
  cv::Size size;                                              ///< Output frame size.
  double fps = 0.0;                                           ///< Output frame rate.
};

/**
 * @brief Open an output writer for the stitched panorama.
 */
VideoOutput openVideoOutput(
    const std::string& path,
    cv::Size size,
    double fps,
    bool preferCuda,
    const std::string& fourcc_str,
    int bitrate_kbps) {
  VideoOutput vo;
  vo.size = size;
  vo.fps = fps;
#if HM_HAS_OPENCV_CUDA_IO
  if (preferCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    try {
      auto codec = cv::cudacodec::Codec::H264;
      vo.gpu = cv::cudacodec::createVideoWriter(path, size, codec, fps, cv::cudacodec::ColorFormat::BGR);
      if (vo.gpu) {
        vo.useCuda = true;
        return vo;
      }
    } catch (const cv::Exception& e) {
      std::cerr << "GPU VideoWriter unavailable, falling back to CPU: " << e.what() << std::endl;
    }
  }
#endif
  int fourcc = 0;
  if (!fourcc_str.empty()) {
    if (fourcc_str.size() == 4) {
      fourcc = cv::VideoWriter::fourcc(fourcc_str[0], fourcc_str[1], fourcc_str[2], fourcc_str[3]);
    } else if (fourcc_str == "H264" || fourcc_str == "h264") {
      fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');
    } else if (fourcc_str == "HEVC" || fourcc_str == "hevc" || fourcc_str == "H265") {
      fourcc = cv::VideoWriter::fourcc('h', 'e', 'v', '1');
    }
  }
  if (fourcc == 0)
    fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
  if (!vo.cpu.open(path, fourcc, fps, size, true)) {
    throw std::runtime_error("Failed to open output writer: " + path);
  }
  vo.useCuda = false;
  (void)bitrate_kbps; // currently unused for CPU writer
  return vo;
}

/**
 * @brief Write a BGR frame to the output using the selected path.
 */
void writeFrame(VideoOutput& out, const cv::Mat& bgr, cv::cuda::Stream& s) {
#if HM_HAS_OPENCV_CUDA_IO
  if (out.useCuda && out.gpu) {
    cv::cuda::GpuMat d;
    d.upload(bgr, s);
    s.waitForCompletion();
    out.gpu->write(d);
  } else {
    out.cpu.write(bgr);
  }
#else
  (void)s;
  out.cpu.write(bgr);
#endif
}

/**
 * @brief Wrap a single image as a batch of repeated references.
 */
std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  std::vector<cv::Mat> batch;
  batch.reserve(batch_size);
  for (int i = 0; i < batch_size; ++i)
    batch.emplace_back(mat);
  return batch;
}

/**
 * @brief Query total frame count from the underlying backend.
 */
int64_t get_total_frames(const VideoInput& in) {
#if HM_HAS_OPENCV_CUDA_IO
  if (in.useCuda && in.reader) {
    double v = 0.0;
    if (in.reader->get(cv::CAP_PROP_FRAME_COUNT, v))
      return v > 0.0 ? static_cast<int64_t>(v + 0.5) : -1;
    return -1;
  }
#endif
  if (!in.useCuda) {
    double v = in.cap.get(cv::CAP_PROP_FRAME_COUNT);
    return v > 0.0 ? static_cast<int64_t>(v + 0.5) : -1;
  }
  return -1;
}

/**
 * @brief Format seconds as H:MM:SS.
 */
static std::string fmt_hhmmss(double seconds) {
  if (seconds < 0)
    return "?";
  int64_t s = static_cast<int64_t>(seconds + 0.5);
  int64_t h = s / 3600;
  int64_t m = (s % 3600) / 60;
  int64_t sec = s % 60;
  std::ostringstream oss;
  oss << h << ':' << std::setw(2) << std::setfill('0') << m << ':' << std::setw(2) << std::setfill('0') << sec;
  return oss.str();
}

/**
 * @brief Periodically print progress with elapsed time and ETA.
 */
static void maybe_print_progress(
    int frame_idx,
    int64_t total_frames,
    const std::chrono::steady_clock::time_point& t0,
    std::chrono::steady_clock::time_point& last_print) {
  using clock = std::chrono::steady_clock;
  const auto now = clock::now();
  const auto since_last = std::chrono::duration_cast<std::chrono::seconds>(now - last_print).count();
  if (since_last < 1 && (frame_idx % 100) != 0)
    return;
  last_print = now;

  const double elapsed = std::chrono::duration<double>(now - t0).count();
  double eta = -1.0;
  double pct = -1.0;
  if (frame_idx > 0 && total_frames > 0) {
    const int64_t remaining = total_frames - frame_idx;
    const double fps_eff = frame_idx / elapsed;
    eta = remaining / std::max(1e-9, fps_eff);
    pct = 100.0 * static_cast<double>(frame_idx) / static_cast<double>(total_frames);
  }
  std::ostringstream msg;
  msg << "Progress: frame " << frame_idx;
  if (total_frames > 0)
    msg << "/" << total_frames;
  if (pct >= 0)
    msg << " (" << std::fixed << std::setprecision(1) << pct << "%)";
  msg << ", elapsed " << fmt_hhmmss(elapsed) << ", ETA " << fmt_hhmmss(eta);
  std::cout << msg.str() << std::endl;
}

} // namespace

/**
 * @brief Program entry point for 3-camera stitcher.
 */
int main(int argc, char** argv) {
  std::string path0;
  std::string path1;
  std::string path2;
  std::string out_path = "stitched_three.mp4";
  std::string control_dir;
  std::string fourcc = "mp4v"; // fallback
  bool prefer_gpu_decode = true; // Prefer NVDEC when available.
  bool prefer_gpu_encode = true; // Prefer NVENC when available.
  bool show = false;
  double show_scale = 0.0; // 0 = no scaled show
  bool adjust_images = false;
  int device_id = 0;
  int num_levels =
#ifdef __aarch64__
      0;
#else
      6;
#endif
  int max_frames = -1;
  int batch_size = 1;
  int bitrate_kbps = 0; // only used by cudacodec

  // Command-line options.
  static struct option long_options[] = {
      {"left", required_argument, 0, 'L'},
      {"middle", required_argument, 0, 'M'},
      {"right", required_argument, 0, 'R'},
      {"video0", required_argument, 0, '0'},
      {"video1", required_argument, 0, '1'},
      {"video2", required_argument, 0, '2'},
      {"output", required_argument, 0, 'o'},
      {"control", required_argument, 0, 'c'},
      {"levels", required_argument, 0, 'l'},
      {"adjust", required_argument, 0, 'a'},
      {"cuda-device", required_argument, 0, 'd'},
      {"gpu-decode", required_argument, 0, 'G'},
      {"gpu-encode", required_argument, 0, 'E'},
      {"fourcc", required_argument, 0, 'f'},
      {"bitrate-kbps", required_argument, 0, 'b'},
      {"max-frames", required_argument, 0, 'm'},
      {"show", no_argument, 0, 's'},
      {"show-scaled", required_argument, 0, 'S'},
      {0, 0, 0, 0}};

  const char* short_opts = "L:M:R:0:1:2:o:c:l:a:d:G:E:f:b:m:sS:";
  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, short_opts, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'L':
        path0 = optarg;
        break;
      case 'M':
        path1 = optarg;
        break;
      case 'R':
        path2 = optarg;
        break;
      case '0':
        path0 = optarg;
        break;
      case '1':
        path1 = optarg;
        break;
      case '2':
        path2 = optarg;
        break;
      case 'o':
        out_path = optarg;
        break;
      case 'c':
        control_dir = optarg;
        break;
      case 'l':
        num_levels = std::atoi(optarg);
        break;
      case 'a':
        adjust_images = !!std::atoi(optarg);
        break;
      case 'd':
        device_id = std::atoi(optarg);
        break;
      case 'G':
        prefer_gpu_decode = !!std::atoi(optarg);
        break;
      case 'E':
        prefer_gpu_encode = !!std::atoi(optarg);
        break;
      case 'f':
        fourcc = optarg;
        break;
      case 'b':
        bitrate_kbps = std::atoi(optarg);
        break;
      case 'm':
        max_frames = std::atoi(optarg);
        break;
      case 's':
        show = true;
        break;
      case 'S':
        show_scale = std::atof(optarg);
        break;
      default:
        break;
    }
  }

  // Basic argument validation; control masks are required for stitching.
  if (path0.empty() || path1.empty() || path2.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --left <v0.mp4> --middle <v1.mp4> --right <v2.mp4> [--output <out.mp4>] --control <dir>"
              << " [--levels N] [--adjust 0|1] [--cuda-device K] [--gpu-decode 0|1] [--gpu-encode 0|1]"
              << " [--fourcc mp4v|avc1|hevc] [--bitrate-kbps N] [--max-frames N] [--show] [--show-scaled F]"
              << std::endl;
    return 2;
  }
  if (control_dir.empty()) {
    std::cerr << "Error: --control <dir> is required (contains mapping_000*.tif and seam_file.png)" << std::endl;
    return 2;
  }

#if !HM_HAS_OPENCV_CUDA_IO
  if (prefer_gpu_decode || prefer_gpu_encode) {
    std::cerr
        << "OpenCV CUDA video I/O modules are not available; forcing --gpu-decode=0 and --gpu-encode=0" << std::endl;
  }
  prefer_gpu_decode = false;
  prefer_gpu_encode = false;
#endif

  // Select CUDA device and create streams.
  cudaSetDevice(device_id);
  cudaStream_t cu_stream;
  cudaStreamCreate(&cu_stream);
  cv::cuda::Stream cv_stream;

  // Load control masks/remaps.
  hm::pano::ControlMasks3 control_masks;
  if (!control_masks.load(control_dir)) {
    std::cerr << "Failed to load control masks from: " << control_dir << std::endl;
    return 1;
  }

  using T_pipeline = uchar4;
  using T_compute = float4;
  hm::pano::cuda::CudaStitchPano3<T_pipeline, T_compute> pano(
      /*batch_size=*/1, num_levels, control_masks, /*match_exposure=*/adjust_images);

  std::cout << "Canvas: " << pano.canvas_width() << "x" << pano.canvas_height() << std::endl;

  // Open inputs and estimate FPS from available metadata.
  VideoInput vin0 = openVideoInput(path0, prefer_gpu_decode);
  VideoInput vin1 = openVideoInput(path1, prefer_gpu_decode);
  VideoInput vin2 = openVideoInput(path2, prefer_gpu_decode);
  const double fps = vin0.fps > 0.0 ? vin0.fps : (vin1.fps > 0.0 ? vin1.fps : (vin2.fps > 0.0 ? vin2.fps : 30.0));

  // Estimate total frames
  const int64_t t0_frames = get_total_frames(vin0);
  const int64_t t1_frames = get_total_frames(vin1);
  const int64_t t2_frames = get_total_frames(vin2);
  int64_t total_frames = -1;
  if (t0_frames > 0 && t1_frames > 0 && t2_frames > 0) {
    total_frames = std::min(t0_frames, std::min(t1_frames, t2_frames));
  } else if (t0_frames > 0 && t1_frames > 0) {
    total_frames = std::min(t0_frames, t1_frames);
  } else if (t0_frames > 0 && t2_frames > 0) {
    total_frames = std::min(t0_frames, t2_frames);
  } else if (t1_frames > 0 && t2_frames > 0) {
    total_frames = std::min(t1_frames, t2_frames);
  } else if (t0_frames > 0) {
    total_frames = t0_frames;
  } else if (t1_frames > 0) {
    total_frames = t1_frames;
  } else if (t2_frames > 0) {
    total_frames = t2_frames;
  }

  // Open output writer using the panorama canvas size and estimated FPS.
  VideoOutput out = openVideoOutput(
      out_path, cv::Size(pano.canvas_width(), pano.canvas_height()), fps, prefer_gpu_encode, fourcc, bitrate_kbps);

  std::unique_ptr<hm::CudaMat<T_pipeline>> canvas =
      std::make_unique<hm::CudaMat<T_pipeline>>(1, pano.canvas_width(), pano.canvas_height());

  int frame_idx = 0; // Number of frames processed so far.
  const auto t0 = std::chrono::steady_clock::now();
  auto last_print = t0;
  cv::Mat bgra0, bgra1, bgra2;
  while (true) {
    // Early exit for smoke testing / partial processing.
    if (max_frames >= 0 && frame_idx >= max_frames)
      break;
    const bool ok0 = readNextBGRA(vin0, bgra0, cv_stream);
    const bool ok1 = readNextBGRA(vin1, bgra1, cv_stream);
    const bool ok2 = readNextBGRA(vin2, bgra2, cv_stream);
    if (!ok0 || !ok1 || !ok2)
      break;

    hm::CudaMat<T_pipeline> in0(as_batch(bgra0, 1));
    hm::CudaMat<T_pipeline> in1(as_batch(bgra1, 1));
    hm::CudaMat<T_pipeline> in2(as_batch(bgra2, 1));

    auto blended = pano.process(in0, in1, in2, cu_stream, std::move(canvas));
    if (!blended.ok()) {
      std::cerr << blended.status().message() << std::endl;
      return blended.status().code();
    }
    canvas = blended.ConsumeValueOrDie();
    cudaStreamSynchronize(cu_stream);

    cv::Mat panoBGRA = canvas->download();
    if (show)
      hm::utils::show_image("stitched3", panoBGRA, /*wait=*/false);
    if (show_scale > 0.0)
      hm::utils::display_scaled_image("stitched3_scaled", panoBGRA, (float)show_scale, /*wait=*/false);
    cv::Mat panoBGR;
    cv::cvtColor(panoBGRA, panoBGR, cv::COLOR_BGRA2BGR);
    writeFrame(out, panoBGR, cv_stream);

    ++frame_idx;
    maybe_print_progress(frame_idx, total_frames, t0, last_print);
  }

  const double total_elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  // Final summary line with total frames (if known) and total elapsed.
  std::cout << "Processed frames: " << frame_idx;
  if (total_frames > 0)
    std::cout << "/" << total_frames;
  std::cout << ", total time " << fmt_hhmmss(total_elapsed) << std::endl;
#if HM_HAS_OPENCV_CUDA_IO
  if (out.useCuda && out.gpu)
    out.gpu->release();
#endif
  canvas.reset();
  cudaStreamDestroy(cu_stream);
  return 0;
}
