/**
 * @file stitch_two_videos.cpp
 * @brief Sample app to stitch two input videos into a panorama using hm::cupano.
 *
 * This application demonstrates a complete I/O loop around the CUDA panorama
 * stitcher. It decodes frames from two videos, stitches them into a single
 * canvas, and encodes the result to an output file. It optionally previews the
 * stitched output.
 *
 * I/O is done via OpenCV. The app attempts to use GPU-accelerated I/O via the
 * cudacodec module when requested and available, falling back to CPU I/O
 * otherwise.
 *
 * Key options (see `--help` output in code):
 *  - --left/--right: input video paths
 *  - --control: directory containing control mask/remap files
 *  - --output: output video path
 *  - --gpu-decode/--gpu-encode: prefer GPU reader/writer (0|1)
 *  - --levels: number of pyramid levels for the stitcher
 *  - --adjust: exposure matching (0|1)
 *  - --cuda-device: CUDA device id
 *  - --max-frames: limit processed frames (useful for smoke tests)
 *  - --show / --show-scaled: optional display
 */

#include "cupano/cuda/cudaStatus.h"
#include "cupano/pano/controlMasks.h"
#include "cupano/pano/cudaMat.h"
#include "cupano/pano/cudaPano.h"
#include "cupano/utils/showImage.h"

#include <cuda_runtime.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudacodec.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
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
 *
 * The object is opened either with OpenCV's CPU `VideoCapture` or the
 * GPU-based `cudacodec::VideoReader` (NVDEC). `fps` and `size` are best-effort
 * hints as some containers lack accurate metadata. `size` is populated by the
 * first decoded frame when not known a priori.
 */
struct VideoInput {
  bool useCuda = false;                                        ///< True if using GPU reader.
  cv::VideoCapture cap;                                        ///< CPU reader.
  cv::Ptr<cv::cudacodec::VideoReader> reader;                  ///< GPU reader.
  double fps = 0.0;                                            ///< Frames per second (best-effort).
  cv::Size size;                                               ///< Frame size, may be empty until first frame.
};

/**
 * @brief Read the next frame and convert to 8-bit BGRA on host memory.
 *
 * On the GPU path, this downloads the decoded frame to CPU and converts to
 * BGRA in the same CUDA stream. On the CPU path, it converts on the host.
 *
 * @param in        Input video state.
 * @param outBGRA   Output host `cv::Mat` with 4 channels (BGRA, 8u).
 * @param s         CUDA stream for GPU conversions.
 * @return true     If a frame was successfully read and converted.
 */
bool readNextBGRA(VideoInput& in, cv::Mat& outBGRA, cv::cuda::Stream& s) {
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
  } else {
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
}

/**
 * @brief Open a video input and prefer the GPU reader when requested.
 *
 * Attempts to create `cudacodec::VideoReader` when `preferCuda` is true and a
 * CUDA device is available. On failure, falls back to `VideoCapture`.
 *
 * @param path        Input video path.
 * @param preferCuda  Prefer GPU-based decoding when available.
 * @return VideoInput Initialized input descriptor.
 */
VideoInput openVideoInput(const std::string& path, bool preferCuda) {
  VideoInput vi;
  vi.size = {};
  vi.fps = 0.0;
  if (preferCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    try {
      vi.reader = cv::cudacodec::createVideoReader(path);
      if (vi.reader) {
        vi.useCuda = true;
        double f = 0.0;
        if (vi.reader->get(cv::CAP_PROP_FPS, f))
          vi.fps = f;
        // size filled on first frame
        return vi;
      }
    } catch (const cv::Exception& e) {
      std::cerr << "Falling back to CPU reader for: " << path << " due to: " << e.what() << std::endl;
    }
  }
  // CPU fallback
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
 *
 * When GPU encoding is selected and available, uses `cudacodec::VideoWriter`
 * (NVENC). Otherwise, falls back to OpenCV's CPU `VideoWriter`.
 */
struct VideoOutput {
  bool useCuda = false;                                       ///< True if using GPU writer.
  cv::VideoWriter cpu;                                        ///< CPU writer.
  cv::Ptr<cv::cudacodec::VideoWriter> gpu;                    ///< GPU writer.
  cv::Size size;                                              ///< Output frame size.
  double fps = 0.0;                                           ///< Output frame rate.
};

/**
 * @brief Open an output writer for the stitched panorama.
 *
 * @param path          Output video path (e.g., mp4).
 * @param size          Output frame size (canvas size).
 * @param fps           Output frame rate.
 * @param preferCuda    Prefer GPU-based encoding when available.
 * @param fourcc_str    FourCC string (e.g., "mp4v", "avc1", "hevc").
 * @param bitrate_kbps  Requested bitrate for GPU writer (if supported).
 * @return VideoOutput  Initialized output writer.
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
  if (preferCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
    try {
      // Use NVENC H264 by default
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
 *
 * Uploads to GPU when using the GPU writer; otherwise writes via CPU.
 *
 * @param out  Output writer descriptor.
 * @param bgr  Input BGR host image.
 * @param s    CUDA stream for GPU uploads.
 */
void writeFrame(VideoOutput& out, const cv::Mat& bgr, cv::cuda::Stream& s) {
  if (out.useCuda && out.gpu) {
    cv::cuda::GpuMat d;
    d.upload(bgr, s);
    s.waitForCompletion();
    out.gpu->write(d);
  } else {
    out.cpu.write(bgr);
  }
}

/**
 * @brief Wrap a single image as a batch of repeated references.
 *
 * The stitcher API is batch-oriented; this helper forms a batch by repeating
 * references to the same `cv::Mat` to satisfy the interface when batch_size>1.
 *
 * @param mat         Input image.
 * @param batch_size  Desired batch size.
 * @return std::vector<cv::Mat> Views with repeated references to `mat`.
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
 *
 * Not all containers/backends report this reliably; on failure returns -1.
 *
 * @param in  Input video descriptor.
 * @return Total frames or -1 when unknown.
 */
int64_t get_total_frames(const VideoInput& in) {
  // Try to query total frame count. Not always available for GPU reader.
  if (in.useCuda && in.reader) {
    double v = 0.0;
    if (in.reader->get(cv::CAP_PROP_FRAME_COUNT, v))
      return v > 0.0 ? static_cast<int64_t>(v + 0.5) : -1;
    return -1;
  }
  if (!in.useCuda) {
    double v = in.cap.get(cv::CAP_PROP_FRAME_COUNT);
    return v > 0.0 ? static_cast<int64_t>(v + 0.5) : -1;
  }
  return -1;
}

/**
 * @brief Format seconds as H:MM:SS.
 * @param seconds  Seconds (negative yields "?").
 * @return Formatted string.
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
 *
 * Prints roughly once per second (and at least every 100 frames). ETA is
 * derived from effective throughput since `t0`.
 *
 * @param frame_idx    1-based number of frames processed.
 * @param total_frames Total frames if known, else -1.
 * @param t0           Start time for elapsed calculation.
 * @param last_print   Timestamp of last print, updated on print.
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
    return; // print roughly once a second or every 100 frames
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
 * @brief Program entry point.
 *
 * Parses command-line options, loads control masks, opens I/O, and runs the
 * stitch loop while printing progress.
 */
int main(int argc, char** argv) {
  std::string left_path;
  std::string right_path;
  std::string out_path = "stitched_two.mp4";
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

  // Command-line options. See README for usage examples.
  static struct option long_options[] = {
      {"left", required_argument, 0, 'L'},
      {"right", required_argument, 0, 'R'},
      {"output", required_argument, 0, 'o'},
      {"control", required_argument, 0, 'c'},
      {"levels", required_argument, 0, 'l'},
      {"adjust", required_argument, 0, 'a'},
      {"cuda-device", required_argument, 0, 'd'},
      {"gpu-decode", required_argument, 0, 'G'}, // 0/1
      {"gpu-encode", required_argument, 0, 'E'}, // 0/1
      {"fourcc", required_argument, 0, 'f'},
      {"bitrate-kbps", required_argument, 0, 'b'},
      {"max-frames", required_argument, 0, 'm'},
      {"show", no_argument, 0, 's'},
      {"show-scaled", required_argument, 0, 'S'},
      {0, 0, 0, 0}};

  const char* short_opts = "L:R:o:c:l:a:d:G:E:f:b:m:sS:";
  int opt, option_index = 0;
  while ((opt = getopt_long(argc, argv, short_opts, long_options, &option_index)) != -1) {
    switch (opt) {
      case 'L':
        left_path = optarg;
        break;
      case 'R':
        right_path = optarg;
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
  if (left_path.empty() || right_path.empty()) {
    std::cerr << "Usage: " << argv[0] << " --left <left.mp4> --right <right.mp4> [--output <out.mp4>] --control <dir>"
              << " [--levels N] [--adjust 0|1] [--cuda-device K] [--gpu-decode 0|1] [--gpu-encode 0|1]"
              << " [--fourcc mp4v|avc1|hevc] [--bitrate-kbps N] [--max-frames N] [--show] [--show-scaled F]"
              << std::endl;
    return 2;
  }

  if (control_dir.empty()) {
    std::cerr << "Error: --control <dir> is required (contains mapping_000*.tif and seam_file.png)" << std::endl;
    return 2;
  }

  // Select CUDA device and create streams for CUDA (stitcher) and OpenCV CUDA (I/O & colorspace).
  cudaSetDevice(device_id);
  cudaStream_t cu_stream;
  cudaStreamCreate(&cu_stream);
  cv::cuda::Stream cv_stream; // for cudacodec and conversions

  // Load control masks/remaps.
  hm::pano::ControlMasks control_masks;
  if (!control_masks.load(control_dir)) {
    std::cerr << "Failed to load control masks from: " << control_dir << std::endl;
    return 1;
  }

  using T_pipeline = uchar4;
  using T_compute = float4;
  hm::pano::cuda::CudaStitchPano<T_pipeline, T_compute> pano(
      /*batch_size=*/1, num_levels, control_masks, /*match_exposure=*/adjust_images);

  std::cout << "Canvas: " << pano.canvas_width() << "x" << pano.canvas_height() << std::endl;

  // Open inputs and estimate FPS from available metadata.
  VideoInput left = openVideoInput(left_path, prefer_gpu_decode);
  VideoInput right = openVideoInput(right_path, prefer_gpu_decode);
  const double fps = (left.fps > 0.0 ? left.fps : (right.fps > 0.0 ? right.fps : 30.0));

  // Estimate total frames
  const int64_t total_left = get_total_frames(left);
  const int64_t total_right = get_total_frames(right);
  int64_t total_frames = -1;
  if (total_left > 0 && total_right > 0)
    total_frames = std::min(total_left, total_right);
  else if (total_left > 0)
    total_frames = total_left;
  else if (total_right > 0)
    total_frames = total_right;

  // Open output writer using the panorama canvas size and estimated FPS.
  VideoOutput out = openVideoOutput(
      out_path, cv::Size(pano.canvas_width(), pano.canvas_height()), fps, prefer_gpu_encode, fourcc, bitrate_kbps);

  std::unique_ptr<hm::CudaMat<T_pipeline>> canvas =
      std::make_unique<hm::CudaMat<T_pipeline>>(1, pano.canvas_width(), pano.canvas_height());

  int frame_idx = 0; // Number of frames processed so far.
  const auto t0 = std::chrono::steady_clock::now();
  auto last_print = t0;
  cv::Mat leftBGRA, rightBGRA;
  while (true) {
    // Early exit for smoke testing / partial processing.
    if (max_frames >= 0 && frame_idx >= max_frames)
      break;
    bool okL = readNextBGRA(left, leftBGRA, cv_stream);
    bool okR = readNextBGRA(right, rightBGRA, cv_stream);
    if (!okL || !okR)
      break;

    hm::CudaMat<T_pipeline> input1(as_batch(leftBGRA, /*batch_size=*/1));
    hm::CudaMat<T_pipeline> input2(as_batch(rightBGRA, /*batch_size=*/1));

    auto blended = pano.process(input1, input2, cu_stream, std::move(canvas));
    if (!blended.ok()) {
      std::cerr << blended.status().message() << std::endl;
      return blended.status().code();
    }
    canvas = blended.ConsumeValueOrDie();
    cudaStreamSynchronize(cu_stream);

    cv::Mat panoBGRA = canvas->download();
    if (show) {
      hm::utils::show_image("stitched", panoBGRA, /*wait=*/false);
    }
    if (show_scale > 0.0) {
      hm::utils::display_scaled_image("stitched_scaled", panoBGRA, (float)show_scale, /*wait=*/false);
    }
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
  if (out.useCuda && out.gpu)
    out.gpu->release();
  canvas.reset();
  cudaStreamDestroy(cu_stream);
  return 0;
}
