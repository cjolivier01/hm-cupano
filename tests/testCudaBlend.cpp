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
#include <mutex>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <opencv4/opencv2/core/hal/interface.h>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/imgcodecs.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

// Function parameters:
//   - image1, image2: the two RGB images to be adjusted.
//   - seam: the seam mask image (CV_8U) that is larger than both images.
//   - N: number of pixels to sample on each side of the seam.
//   - topLeft1, topLeft2: the (x,y) coordinates (relative to the seam mask)
//                           of the top left corners of image1 and image2, respectively.
void matchSeamImages(
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam,
    int N,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2) {
  // Ensure the seam mask is of type CV_8U.
  if (seam.type() != CV_8U) {
    std::cerr << "Error: Seam mask must be of type CV_8U." << std::endl;
    return;
  }

  // Accumulators for summing per-channel pixel values.
  cv::Scalar sumLeft(0, 0, 0); // For image1 samples (left of the seam).
  cv::Scalar sumRight(0, 0, 0); // For image2 samples (right of the seam).
  int countLeft = 0, countRight = 0;

  // ----- Process image1 (sampling from the left side of the seam) -----
  // Only examine the middle 50% of image1's rows.
  int startRow1 = image1.rows / 4;
  int endRow1 = (3 * image1.rows) / 4;
  for (int r = startRow1; r < endRow1; r++) {
    // Map image1’s local row (r) to the seam mask’s row coordinate.
    int globalRow = topLeft1.y + r;
    if (globalRow < 0 || globalRow >= seam.rows)
      continue; // Row is outside the seam mask.

    // Define the horizontal span of image1 in the seam mask.
    int colStart = topLeft1.x;
    int colEnd = topLeft1.x + image1.cols;

    // Find the seam boundary: the first column (within image1’s span)
    // where the seam mask pixel equals 1.
    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols)
        continue;
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1)
      continue; // No seam boundary found for this row.

    // Convert the global seam column to image1’s local coordinate.
    int seamLocalCol = seamGlobalCol - topLeft1.x;

    // Sample up to N pixels immediately to the left of the seam boundary.
    int sampleStart = std::max(0, seamLocalCol - N);
    for (int c = sampleStart; c < seamLocalCol; c++) {
      // Safety check.
      if (c < 0 || c >= image1.cols)
        continue;

      // Depending on the image depth, read the pixel appropriately.
      if (image1.depth() == CV_8U) {
        cv::Vec3b pixel = image1.at<cv::Vec3b>(r, c);
        sumLeft[0] += pixel[0];
        sumLeft[1] += pixel[1];
        sumLeft[2] += pixel[2];
      } else if (image1.depth() == CV_32F) {
        cv::Vec3f pixel = image1.at<cv::Vec3f>(r, c);
        sumLeft[0] += pixel[0];
        sumLeft[1] += pixel[1];
        sumLeft[2] += pixel[2];
      }
      countLeft++;
    }
  }

  // ----- Process image2 (sampling from the right side of the seam) -----
  // Only examine the middle 50% of image2's rows.
  int startRow2 = image2.rows / 4;
  int endRow2 = (3 * image2.rows) / 4;
  for (int r = startRow2; r < endRow2; r++) {
    // Map image2’s local row (r) to the seam mask’s row coordinate.
    int globalRow = topLeft2.y + r;
    if (globalRow < 0 || globalRow >= seam.rows)
      continue;

    // Define the horizontal span of image2 in the seam mask.
    int colStart = topLeft2.x;
    int colEnd = topLeft2.x + image2.cols;

    // Find the seam boundary in image2’s region of the seam mask.
    int seamGlobalCol = -1;
    for (int c = colStart; c < colEnd; c++) {
      if (c < 0 || c >= seam.cols)
        continue;
      if (seam.at<uchar>(globalRow, c) == 1) {
        seamGlobalCol = c;
        break;
      }
    }
    if (seamGlobalCol == -1)
      continue; // No seam boundary found in this row.

    // Convert the global seam column to image2’s local coordinate.
    int seamLocalCol = seamGlobalCol - topLeft2.x;

    // Sample up to N pixels immediately to the right of the seam boundary.
    int sampleEnd = std::min(image2.cols, seamLocalCol + N);
    for (int c = seamLocalCol; c < sampleEnd; c++) {
      if (c < 0 || c >= image2.cols)
        continue;

      if (image2.depth() == CV_8U) {
        cv::Vec3b pixel = image2.at<cv::Vec3b>(r, c);
        sumRight[0] += pixel[0];
        sumRight[1] += pixel[1];
        sumRight[2] += pixel[2];
      } else if (image2.depth() == CV_32F) {
        cv::Vec3f pixel = image2.at<cv::Vec3f>(r, c);
        sumRight[0] += pixel[0];
        sumRight[1] += pixel[1];
        sumRight[2] += pixel[2];
      }
      countRight++;
    }
  }

  // Check that we have collected samples from both images.
  if (countLeft == 0 || countRight == 0) {
    std::cerr << "Error: Not enough seam samples collected for adjustment." << std::endl;
    return;
  }

  // Compute per-channel averages.
  cv::Scalar avgLeft = sumLeft * (1.0 / countLeft);
  cv::Scalar avgRight = sumRight * (1.0 / countRight);
  std::cout << "Average values (Image1, left side): " << avgLeft << std::endl;
  std::cout << "Average values (Image2, right side): " << avgRight << std::endl;

  // Compute an offset per channel (half the difference).
  // The idea is to subtract this offset from image1 and add it to image2.
  cv::Scalar offset = (avgLeft - avgRight) * 0.5;
  std::cout << "Offset: " << offset << std::endl;

  // Helper lambda: adjusts an image by a per-channel amount.
  auto adjustImage = [&](cv::Mat& img, cv::Scalar adjustment) {
    if (img.depth() == CV_8U) {
      cv::Mat floatImg;
      img.convertTo(floatImg, CV_32F);
      std::vector<cv::Mat> channels;
      cv::split(floatImg, channels);
      for (int i = 0; i < 3; i++) {
        channels[i] += static_cast<float>(adjustment[i]);
      }
      cv::merge(channels, floatImg);
      // Clamp the adjusted values to the valid range [0,255].
      cv::min(floatImg, 255.0, floatImg);
      cv::max(floatImg, 0.0, floatImg);
      floatImg.convertTo(img, CV_8U);
    } else if (img.depth() == CV_32F) {
      std::vector<cv::Mat> channels;
      cv::split(img, channels);
      for (int i = 0; i < 3; i++) {
        channels[i] += static_cast<float>(adjustment[i]);
      }
      cv::merge(channels, img);
    }
  };

  // Adjust the images: subtract the offset from image1 and add it to image2.
  adjustImage(image1, -offset);
  adjustImage(image2, offset);
}

std::vector<cv::Mat> as_batch(const cv::Mat& mat, int batch_size) {
  return std::vector<cv::Mat>(batch_size, mat);
}

int main(int argc, char** argv) {
  // Usage check.
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <game-id>" << std::endl;
    return -1;
  }

  cudaSetDevice(0);
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::string game_id = argv[1];
  std::string game_dir = std::string(::getenv("HOME")) + "/Videos/" + game_id + "/";

  // stitch-fix
  // Left video frame
  std::string sample_img_left_path = game_dir + "GX010100.png";
  // Right video frame
  std::string sample_img_right_path = game_dir + "GX010019.png";

  // std::string sample_img_left_path = game_dir + "GX010097.png";
  // std::string sample_img_right_path = game_dir + "GX010016.png";

  // PDP
  // std::string sample_img_left_path = game_dir + "GX010087.png";
  // std::string sample_img_right_path = game_dir + "GX010003.png";

  cv::Mat sample_img_left = cv::imread(sample_img_left_path, cv::IMREAD_COLOR);
  assert(!sample_img_left.empty());
  cv::Mat sample_img_right = cv::imread(sample_img_right_path, cv::IMREAD_COLOR);
  assert(!sample_img_right.empty());

  hm::pano::ControlMasks control_masks;
  control_masks.load(game_dir);

  // Compute canvas size
  const int canvas_width = std::max(
      control_masks.positions[0].xpos + control_masks.img1_col.cols,
      control_masks.positions[1].xpos + control_masks.img2_col.cols);
  const int canvas_height = std::max(
      control_masks.positions[0].ypos + control_masks.img1_col.rows,
      control_masks.positions[1].ypos + control_masks.img2_col.rows);
  std::cout << "Canvas size: " << canvas_width << " x " << canvas_height << std::endl;

// Configurable parameter: number of pyramid levels.
#ifdef __aarch64__
  // Lower compute, quick and dirty
  int numLevels = 0;
  // int numLevels = 6;
#else
  int numLevels = 6;
  // int numLevels = 2;
  // int numLevels = 6;
  // int numLevels = 0;
#endif

#if 1
#if 1
  using T = uchar3;
  // using T_compute = uchar3;
  using T_compute = float3;
  // using T_compute = half3;
#else
  using T = float3;
  using T_compute = float3;
#endif
#else
  using T = float;
  using T_compute = __half;
#endif

  const int CV_T_PIPELINE = cudaPixelTypeToCvType(CudaTypeToPixelType<T>::value);
  const int CV_T_COMPUTE3 = cudaPixelTypeToCvType(CudaTypeToPixelType<T_compute>::value);

  if (std::is_floating_point_v<BaseScalar_t<T>>) {
    sample_img_left.convertTo(sample_img_left, CV_T_PIPELINE, 1.0 / 255.0);
    sample_img_right.convertTo(sample_img_right, CV_T_PIPELINE, 1.0 / 255.0);
  }

  constexpr int kBatchSize = 1;
  // constexpr int kBatchSize = 2;

  hm::pano::cuda::StitchingContext<T, T_compute> stitch_context(
      /*batch_size=*/kBatchSize, /*is_hard_seam=*/numLevels == 0);

  //
  // CanvasManager
  //
  hm::pano::CanvasManager canvas_manager(
      hm::pano::CanvasInfo{
          .width = canvas_width,
          .height = canvas_height,
          .positions =
              {cv::Point(control_masks.positions[0].xpos, control_masks.positions[0].ypos),
               cv::Point(control_masks.positions[1].xpos, control_masks.positions[1].ypos)}},
      /*minimize_blend=*/!stitch_context.is_hard_seam());
  canvas_manager._remapper_1.width = control_masks.img1_col.cols;
  canvas_manager._remapper_1.height = control_masks.img1_col.rows;
  canvas_manager._remapper_2.width = control_masks.img2_col.cols;
  canvas_manager._remapper_2.height = control_masks.img2_col.rows;

  canvas_manager.updateMinimizeBlend(control_masks.img1_col.size(), control_masks.img2_col.size());

  cv::Mat blend_seam = canvas_manager.convertMaskMat(control_masks.whole_seam_mask_image);
  assert(!blend_seam.empty());
  blend_seam = blend_seam.clone();

  auto canvas = std::make_unique<CudaMat<T>>(
      stitch_context.batch_size(), canvas_manager.canvas_width(), canvas_manager.canvas_height());

  assert(control_masks.img1_col.type() == CV_16U);
  stitch_context.remap_1_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_col);
  stitch_context.remap_1_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img1_row);

  stitch_context.remap_2_x = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_col);
  stitch_context.remap_2_y = std::make_unique<CudaMat<uint16_t>>(control_masks.img2_row);

  if (!stitch_context.is_hard_seam()) {
    blend_seam.convertTo(blend_seam, CV_T_COMPUTE3);
    stitch_context.cudaFull1 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);
    stitch_context.cudaFull2 =
        std::make_unique<CudaMat<T_compute>>(stitch_context.batch_size(), blend_seam.cols, blend_seam.rows);

    stitch_context.cudaBlendSoftSeam = std::make_unique<CudaMat<T_compute>>(blend_seam);
    stitch_context.laplacian_blend_context = std::make_unique<CudaBatchLaplacianBlendContext<BaseScalar_t<T_compute>>>(
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        numLevels,
        /*batch_size=*/stitch_context.batch_size());
  } else {
    assert(blend_seam.type() == CV_8U);
    stitch_context.cudaBlendHardSeam = std::make_unique<CudaMat<unsigned char>>(blend_seam);
  }

  //
  // The actual incoming images
  //

  // matchSeamImages(
  //     sample_img_left,
  //     sample_img_left,
  //     control_masks.whole_seam_mask_image,
  //     /*N=*/10,
  //     cv::Point(canvas_manager.canvas_info_.positions[0].x, canvas_manager.canvas_info_.positions[0].y),
  //     cv::Point(canvas_manager.canvas_info_.positions[1].x, canvas_manager.canvas_info_.positions[1].y));

  // assert(sample_img_left.type() == CV_8UC3);
  // assert(sample_img_right.type() == CV_8UC3);

  CudaMat<T> sampleImage1(as_batch(sample_img_left, kBatchSize));
  CudaMat<T> sampleImage2(as_batch(sample_img_right, kBatchSize));

  auto blendedCanvasResult = hm::pano::cuda::CudaStitchPano<T, T_compute>::process(
      sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(canvas));
  if (!blendedCanvasResult.ok()) {
    std::cerr << blendedCanvasResult.status().message() << std::endl;
    return blendedCanvasResult.status().code();
  }
  auto blendedCanvas = blendedCanvasResult.ConsumeValueOrDie();
  // SHOW_SMALL(blendedCanvas);
  //  SHOW_IMAGE(blendedCanvas);
  // SHOW_SCALED(blendedCanvas, 0.25);

  // blendedCanvas = process(sampleImage1, sampleImage2, stitch_context, canvas_manager, stream);

  // cudaStreamSynchronize(stream);

  // display.render("cudaBlendedFull", CudaSurface(cudaBlendedFull), stream);

#if 1 /* perf test */
  auto start_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();

  size_t frame_count = 100;
  for (size_t i = 0; i < frame_count; ++i) {
    blendedCanvas = hm::pano::cuda::CudaStitchPano<T, T_compute>::process(
                        sampleImage1, sampleImage2, stitch_context, canvas_manager, stream, std::move(blendedCanvas))
                        .ConsumeValueOrDie();
    cudaStreamSynchronize(stream);
  }

  auto stop_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())
          .count();
  float ms = stop_ms - start_ms;
  float sec_per_frame = (ms / 1000) / (frame_count * stitch_context.batch_size());
  std::cout << "Blend speed: " << (1.0 / sec_per_frame) << "fps" << std::endl;
#endif

  cudaStreamDestroy(stream);

  return cudaSuccess;
}
