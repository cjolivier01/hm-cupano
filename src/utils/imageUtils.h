#pragma once

#include <opencv2/opencv.hpp>

namespace hm {
namespace utils {

/**
 * @brief Get the minimum and maximum pixel values for each channel in a multi‐channel cv::Mat.
 *
 * This function splits the image into its separate channels, then applies cv::minMaxLoc
 * on each channel independently.
 *
 * @param mat   Input image (must be non‐empty and have at least two channels).
 * @return      A vector of (min, max) pairs, one for each channel, in order.
 *
 * @throws std::invalid_argument if mat is empty or has only one channel.
 */
std::vector<std::pair<double, double>> getMinMaxPerChannel(const cv::Mat& mat);

// Clamp all pixel values in-place to [minVal, maxVal]
void clamp(cv::Mat& img, float minVal = 0.0f, float maxVal = 255.0f);

// Linearly stretch/squish all pixel values in-place so that
// the overall min→max range maps to [lo, hi]
void stretch(cv::Mat& img, float lo = 0.0f, float hi = 255.0f);

// Convert to uchar with clamping
cv::Mat convert_to_uchar(cv::Mat image);

} // namespace utils
} // namespace hm
