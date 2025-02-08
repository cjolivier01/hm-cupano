#include "cudaPano.h"

namespace hm {
namespace pano {

//
// For hard seams, the seam visibility can be reduced significantly
// by matching the colorspace of the images.  We do this by
// analyzing a few pixels on either side of the seam and vertically
// in the middle and then compute +-offsets for the images to help
// them match.
//
// Function parameters:
//   - image1, image2: the two RGB images to be adjusted.
//   - seam: the seam mask image (CV_8U) that is larger than both images.
//   - N: number of pixels to sample on each side of the seam.
//   - topLeft1, topLeft2: the (x,y) coordinates (relative to the seam mask)
//                           of the top left corners of image1 and image2,
//                           respectively.
std::optional<cv::Scalar> match_seam_images(
    cv::Mat& image1,
    cv::Mat& image2,
    const cv::Mat& seam,
    int N,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose) {
  // Ensure the seam mask is of type CV_8U.
  if (seam.type() != CV_8U) {
    std::cerr << "Error: Seam mask must be of type CV_8U." << std::endl;
    return std::nullopt;
  }

  // Accumulators for summing per-channel pixel values.
  cv::Scalar sumLeft(0, 0, 0); // For image1 samples (left of the seam).
  cv::Scalar sumRight(0, 0, 0); // For image2 samples (right of the seam).
  int countLeft = 0, countRight = 0;

  std::vector<int> seam_column(seam.rows, -1);

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
      if (seam.at<uchar>(globalRow, c) == 0) {
        seamGlobalCol = c;
        seam_column[r] = seamGlobalCol;
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
      if (seam.at<uchar>(globalRow, c) == 0) {
        seamGlobalCol = c;
        // assert(seamGlobalCol == seam_column[r] || seam_column[r] == -1);
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
    return std::nullopt;
  }

  // Compute per-channel averages.
  cv::Scalar avgLeft = sumLeft * (1.0 / countLeft);
  cv::Scalar avgRight = sumRight * (1.0 / countRight);
  if (verbose) {
    std::cout << "Average values (Image1, left side): " << avgLeft << std::endl;
    std::cout << "Average values (Image2, right side): " << avgRight << std::endl;
  }
  // Compute an offset per channel (half the difference).
  // The idea is to subtract this offset from image1 and add it to image2.
  cv::Scalar offset = (avgLeft - avgRight) * 0.5;
  if (verbose) {
    std::cout << "Offset: " << offset << std::endl;
  }

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
      img = img.clone();
    }
  };

  // Adjust the images: subtract the offset from image1 and add it to image2.
  adjustImage(image1, -offset);
  adjustImage(image2, offset);
  return offset;
}

namespace cuda {} // namespace cuda
} // namespace pano
} // namespace hm
