#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include "cupano/cuda/cudaTypes.h"

namespace hm {
namespace utils {

void show_image(
    const std::string& label,
    const cv::Mat& img,
    bool wait = true,
    float scale = 0.0f,
    bool squish = false);
void display_scaled_image(
    const std::string& label,
    cv::Mat image,
    float scale = 1.0,
    bool wait = true,
    bool squish = false);

template <typename PIXEL_T>
void show_surface(const std::string& label, const CudaSurface<PIXEL_T>& surface, bool wait);
bool destroy_surface_window();

std::pair<double, double> get_min_max(const cv::Mat& mat);
cv::Mat make_fake_mask_like(const cv::Mat& mask);

#define SHOW_SURFACE(_mat$)                                                            \
  do {                                                                                 \
    ::hm::utils::show_surface(std::string(#_mat$), (_mat$)->surface(), /*wait=*/true); \
  } while (false)

#define SHOW_IMAGE(_mat$)                                                             \
  do {                                                                                \
    ::hm::utils::show_image(std::string(#_mat$), (_mat$)->download(), /*wait=*/true); \
  } while (false)

#define SHOW_SCALED(_mat$, _scale$)                                                                      \
  do {                                                                                                   \
    ::hm::utils::display_scaled_image(std::string(#_mat$), (_mat$)->download(), _scale$, /*wait=*/true); \
  } while (false)

#define SHOW_SMALL(_mat$)     \
  do {                        \
    SHOW_SCALED(_mat$, 0.05); \
  } while (false)

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

} // namespace utils
} // namespace hm
