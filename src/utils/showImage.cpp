#include "cupano/utils/showImage.h"
#include "cupano/utils/cudaGLWindow.h"

#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <unordered_set>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

namespace hm {
namespace utils {

namespace {
thread_local std::unique_ptr<CudaGLWindow> gl_window;

CudaGLWindow* get_gl_window(int w, int h, int channels, const char* title) {
  if (!gl_window) {
    gl_window = std::make_unique<CudaGLWindow>(w, h, channels, title);
  }
  return gl_window.get();
}

} // namespace

void warn_cpu_image_preview_unavailable() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::cerr << "CPU image preview is unavailable because the OpenCV GUI module is not linked; continuing headless."
              << std::endl;
  });
}

int kbhit() {
  struct termios oldt, newt;
  int ch;
  int oldf;

  tcgetattr(STDIN_FILENO, &oldt);
  newt = oldt;
  newt.c_lflag &= ~(ICANON | ECHO);
  tcsetattr(STDIN_FILENO, TCSANOW, &newt);
  oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
  fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

  ch = getchar();

  tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
  fcntl(STDIN_FILENO, F_SETFL, oldf);

  if (ch != EOF) {
    ungetc(ch, stdin);
    return 1;
  }

  return 0;
}

int wait_key(CudaGLWindow* window = nullptr) {
  int c;
  while (!(c = kbhit())) {
    if (window && window->isKeyPressed(GLFW_KEY_ESCAPE)) {
      // ESCAPE?
      constexpr int kEscapeKey = 27;
      return kEscapeKey;
    }
    usleep(100);
  }
  return c;
}

void show_image(const std::string& label, const cv::Mat& img, bool wait, float scale, bool squish) {
  warn_cpu_image_preview_unavailable();
  (void)label;
  (void)img;
  (void)wait;
  (void)scale;
  (void)squish;
}

template <typename PIXEL_T>
void show_surface(const std::string& label, const CudaSurface<PIXEL_T>& surface, bool wait) {
  CudaGLWindow* gl_window =
      get_gl_window(surface.width, surface.height, sizeof(PIXEL_T) / sizeof(PIXEL_T::x), label.c_str());
  if (!gl_window) {
    return;
  }
  gl_window->render(surface);
  if (wait) {
    wait_key(gl_window);
  }
}

template void show_surface<uchar3>(const std::string& label, const CudaSurface<uchar3>& surface, bool wait);
template void show_surface<float3>(const std::string& label, const CudaSurface<float3>& surface, bool wait);
template void show_surface<uchar4>(const std::string& label, const CudaSurface<uchar4>& surface, bool wait);

bool destroy_surface_window() {
  if (!gl_window) {
    return false;
  }
  gl_window.reset();
  return true;
}

void display_scaled_image(const std::string& label, cv::Mat image, float scale, bool wait, bool squish) {
  warn_cpu_image_preview_unavailable();
  (void)label;
  (void)image;
  (void)scale;
  (void)wait;
  (void)squish;
}

std::pair<double, double> get_min_max(const cv::Mat& mat) {
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;

  // Get the minimum and maximum values and their locations
  cv::minMaxLoc(mat, &minVal, &maxVal, &minLoc, &maxLoc);
  return std::make_pair(minVal, maxVal);
}

template <typename T>
std::set<T> get_unique_values(const cv::Mat& mat, const std::unordered_set<T>& ignore = {}) {
  std::set<T> unique_values;

  // Check if the data type of the matrix matches the template type
  if (mat.type() != cv::DataType<T>::type) {
    throw std::invalid_argument("Matrix data type does not match the template type T");
  }

  // Iterate over each element in the matrix
  for (int i = 0; i < mat.rows; ++i) {
    for (int j = 0; j < mat.cols; ++j) {
      T value = mat.at<T>(i, j);
      // Add to set if not in ignore set
      if (ignore.find(value) == ignore.end()) {
        unique_values.insert(value);
      }
    }
  }

  return unique_values;
}

} // namespace utils
} // namespace hm
