#pragma once

#include <opencv2/opencv.hpp>
#include <cstdint>

namespace hm {
namespace cv_type_traits {

// Primary template: undefined (will trigger a static_assert if used with an unsupported type)
template <typename T>
struct CVType {
  static_assert(sizeof(T) == 0, "CVType<T>: unsupported type, no OpenCV depth available");
};

// Specializations for standard integer and float types:

template <>
struct CVType<unsigned char> {
  static constexpr int value = CV_8U;
};
template <>
struct CVType<signed char> {
  static constexpr int value = CV_8S;
};
template <>
struct CVType<uint16_t> {
  static constexpr int value = CV_16U;
};
template <>
struct CVType<int16_t> {
  static constexpr int value = CV_16S;
};
template <>
struct CVType<int32_t> {
  static constexpr int value = CV_32S;
};
template <>
struct CVType<uint32_t> {
  static constexpr int value = CV_32S; /* no CV_32U in OpenCV; 32S is closest */
};
template <>
struct CVType<float> {
  static constexpr int value = CV_32F;
};
template <>
struct CVType<double> {
  static constexpr int value = CV_64F;
};

// If you also use `short` or `int` directly:
template <>
struct CVType<long> {
  static constexpr int value =
      CV_32S; /* long typically 32-bit on Windows, 64-bit on Linux—choose whichever suits your build. */
};
// template <>
// struct CVType<long long> {
//   static constexpr int value = CV_64F; /* no 64S-depth macros; map to double */
// };

// Half-precision (if OpenCV has CV_16F defined, available since OpenCV 4.5+):
#ifdef CV_16F
template <>
struct CVType<cv::float16_t> {
  static constexpr int value = CV_16F;
};
// Some compilers use __fp16 or _Float16 instead of cv::float16_t:
#if defined(__FLT16_TYPE__) || defined(__fp16) || defined(_Float16)
template <>
struct CVType<__fp16> {
  static constexpr int value = CV_16F;
};
template <>
struct CVType<_Float16> {
  static constexpr int value = CV_16F;
};
#endif
#endif // CV_16F

// Convenience alias:
template <typename T>
static constexpr int CVDepth = CVType<T>::value;

} // namespace cv_type_traits
} // namespace hm
