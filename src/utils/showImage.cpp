#include "cupano/utils/showImage.h"
#include "cupano/utils/imageUtils.h"
#include "cupano/utils/cudaGLWindow.h"

#include <set>
#include <unordered_set>

#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <fcntl.h>
#include <opencv4/opencv2/imgproc.hpp>
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
  if (scale != 0 && scale != 1) {
    cv::Size newSize(static_cast<int>(scale * (img.cols + 0.5f)), static_cast<int>(scale * (img.rows + 0.5f)));
    cv::Mat dest;
    if (scale < 1) {
      cv::resize(img, dest, newSize, 0.0, 0.0, cv::INTER_NEAREST);
    } else {
      cv::resize(img, dest, newSize, 0.0, 0.0, cv::INTER_NEAREST/*cv::INTER_LINEAR*/);
    }
    cv::imshow(label, convert_to_uchar(std::move(dest)));
  } else {
    cv::imshow(label, convert_to_uchar(img.clone()));
  }
  cv::waitKey(wait ? 0 : 1);
}

template <typename PIXEL_T>
void show_surface(const std::string& label, const CudaSurface<PIXEL_T>& surface, bool wait) {
  CudaGLWindow* gl_window = get_gl_window(surface.width, surface.height, sizeof(PIXEL_T), label.c_str());
  if (!gl_window) {
    return;
  }
  gl_window->render(surface);
  if (wait) {
    wait_key(gl_window);
  }
}

template void show_surface<uchar3>(const std::string& label, const CudaSurface<uchar3>& surface, bool wait);
template void show_surface<uchar4>(const std::string& label, const CudaSurface<uchar4>& surface, bool wait);

bool destroy_surface_window() {
  if (!gl_window) {
    return false;
  }
  gl_window.reset();
  return true;
}

void display_scaled_image(const std::string& label, cv::Mat image, float scale, bool wait, bool squish) {
  if (scale != 1.0f) {
    // Calculate new dimensions
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);

    // Resize the image
    cv::resize(image, image, cv::Size(newWidth, newHeight));
  }
  if (squish) {
    stretch(image, 0.0f, 255.0f);
  }
  // Display the image
  cv::imshow(label, convert_to_uchar(image));
  cv::waitKey(wait ? 0 : 1); // Wait for a keystroke in the window
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
