#include "cupano/utils/imageUtils.h"

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace hm {
namespace utils {

void set_alpha_pixels(cv::Mat& image, const cv::Vec3b& color) {
  // Check that the image is non-empty and has 4 channels.
  if (image.empty() || image.channels() != 4) {
    return;
  }

  // Iterate over each row.
  for (int y = 0; y < image.rows; y++) {
    // Get pointer to the beginning of row 'y'. Each pixel is a Vec4b.
    cv::Vec4b* rowPtr = image.ptr<cv::Vec4b>(y);
    for (int x = 0; x < image.cols; x++) {
      // Check if alpha channel is 0.
      if (rowPtr[x][3] == 0) {
        // Set B, G, R channels to the specified color.
        rowPtr[x][0] = color[0]; // Blue channel.
        rowPtr[x][1] = color[1]; // Green channel.
        rowPtr[x][2] = color[2]; // Red channel.
        rowPtr[x][2] = 255;
      }
    }
  }
}

cv::Mat convert_to_uchar(cv::Mat image) {
  // Check if the image is of a floating-point type
  if (image.depth() == CV_16F || image.depth() == CV_32F || image.depth() == CV_64F) {
    cv::Mat ucharImage;
    // convertTo automatically applies saturate_cast, clamping values to [0, 255]
    image.convertTo(ucharImage, CV_8U);
    set_alpha_pixels(ucharImage, {255, 0, 0});
    return ucharImage;
  }
  // For non-floating point images, return a copy (or handle as needed)
  set_alpha_pixels(image, {255, 0, 0});
  return image;
}

cv::Mat make_fake_mask_like(const cv::Mat& mask) {
  cv::Mat img(mask.rows, mask.cols, CV_32FC1, cv::Scalar(0));

  // Define a region of interest (ROI) for the left half of the image.
  cv::Rect leftHalfROI(0, 0, mask.cols / 2, mask.rows);

  // Set all pixels in the left half to 1.
  img(leftHalfROI).setTo(1.0f);
  return img;
}

std::vector<std::pair<double, double>> getMinMaxPerChannel(const cv::Mat& mat) {
  if (mat.empty()) {
    throw std::invalid_argument("getMinMaxPerChannel: Input cv::Mat is empty.");
  }
  int nChannels = mat.channels();
  if (nChannels < 2) {
    throw std::invalid_argument("getMinMaxPerChannel: Input must have multiple channels.");
  }

  // Split into individual channels
  std::vector<cv::Mat> channels;
  cv::split(mat, channels);

  // Prepare a vector to hold (min,max) for each channel
  std::vector<std::pair<double, double>> minsAndMaxs;
  minsAndMaxs.reserve(nChannels);

  for (int c = 0; c < nChannels; ++c) {
    double minVal = 0.0, maxVal = 0.0;
    cv::minMaxLoc(channels[c], &minVal, &maxVal);
    minsAndMaxs.emplace_back(minVal, maxVal);
  }
  return minsAndMaxs;
}

// Clamp all pixel values in-place to [minVal, maxVal]
void clamp(cv::Mat& img, float minVal, float maxVal) {
  CV_Assert(img.type() == CV_32FC3);
  cv::Scalar sMin(minVal, minVal, minVal);
  cv::Scalar sMax(maxVal, maxVal, maxVal);
  // first clamp below
  cv::max(img, sMin, img);
  // then clamp above
  cv::min(img, sMax, img);
}

// Linearly stretch/squish all pixel values in-place so that
// the overall min→max range maps to [lo, hi]
void stretch(cv::Mat& img, float lo, float hi) {
  CV_Assert(!img.empty());

  int type = img.type();

  if (type == CV_32FC3) {
    // Already in float32
  } else if (type == CV_8UC3 || type == CV_8UC4) {
    // Convert to float and normalize to [0, 1]
    img.convertTo(img, CV_32F, 1.0 / 255.0);
    if (type == CV_8UC3) img = img.reshape(3); // Ensure 3 channels
    else if (type == CV_8UC4) img = img.reshape(4);
  } else {
    CV_Error(cv::Error::StsUnsupportedFormat, "Unsupported image type in stretch()");
  }

  // flatten to single-channel view for min/max computation
  cv::Mat flat = img.reshape(1);
  double imgMin, imgMax;
  cv::minMaxLoc(flat, &imgMin, &imgMax);

  // if constant image, set to midpoint
  if (imgMin == imgMax) {
    float mid = 0.5f * (lo + hi);
    img.setTo(cv::Scalar(mid, mid, mid, mid));
    return;
  }

  // compute scale and bias
  float scale = (hi - lo) / static_cast<float>(imgMax - imgMin);
  cv::Scalar sMin(imgMin, imgMin, imgMin, imgMin);
  cv::Scalar sLo(lo, lo, lo, lo);

  // (img - imgMin) * scale + lo
  cv::subtract(img, sMin, img);    // img -= imgMin
  cv::multiply(img, scale, img);   // img *= scale
  cv::add(img, sLo, img);          // img += lo
}


} // namespace utils
} // namespace hm
