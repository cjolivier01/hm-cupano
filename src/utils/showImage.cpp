#include "cupano/utils/showImage.h"
#include "cupano/utils/cudaGLWindow.h"
#include "cupano/utils/imageUtils.h"
#include "jetson-utils/display/glDisplay.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <fcntl.h>
#include <opencv2/imgproc.hpp>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

namespace hm {
namespace utils {

int kbhit();

namespace {
thread_local std::unique_ptr<CudaGLWindow> gl_window;
thread_local std::unordered_map<std::string, std::unique_ptr<glDisplay>> cpu_image_windows;

CudaGLWindow* get_gl_window(int w, int h, int channels, const char* title) {
  if (!gl_window) {
    gl_window = std::make_unique<CudaGLWindow>(w, h, channels, title);
  }
  return gl_window.get();
}

bool preview_key_handler(uint16_t event, int a, int b, void* user) {
  auto* pressed = static_cast<bool*>(user);
  if (!pressed) {
    return false;
  }
  if ((event == KEY_STATE || event == KEY_MODIFIED) && b == KEY_PRESSED) {
    *pressed = true;
  } else if (event == KEY_CHAR) {
    (void)a;
    *pressed = true;
  } else if (event == WINDOW_CLOSED) {
    *pressed = true;
  }
  return false;
}

void wait_for_preview_key(glDisplay* display) {
  if (!display) {
    return;
  }
  bool pressed = false;
  display->AddEventHandler(preview_key_handler, &pressed);
  while (display->IsOpen() && !pressed && !kbhit()) {
    display->ProcessEvents();
    usleep(1000);
  }
  display->RemoveEventHandler(preview_key_handler, &pressed);
}

glDisplay* get_cpu_image_window(const std::string& label, int width, int height) {
  auto& window = cpu_image_windows[label];
  if (!window || window->IsClosed()) {
    videoOptions options;
    options.width = width;
    options.height = height;
    window.reset(glDisplay::Create(options));
    if (window) {
      window->SetTitle(label.c_str());
    }
  }
  return window.get();
}

cv::Mat resize_for_preview(const cv::Mat& image, float scale) {
  if (scale == 0.0f || scale == 1.0f) {
    return image.clone();
  }
  const int width = std::max(1, static_cast<int>(scale * (image.cols + 0.5f)));
  const int height = std::max(1, static_cast<int>(scale * (image.rows + 0.5f)));
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(width, height), 0.0, 0.0, cv::INTER_NEAREST);
  return resized;
}

cv::Mat prepare_cpu_preview_image(cv::Mat image, bool squish, imageFormat* format) {
  if (image.empty()) {
    throw std::invalid_argument("cannot show an empty image");
  }
  if (squish) {
    stretch(image, 0.0f, 255.0f);
  }
  cv::Mat uchar_image = convert_to_uchar(std::move(image));
  cv::Mat render_image;
  switch (uchar_image.channels()) {
    case 1:
      cv::cvtColor(uchar_image, render_image, cv::COLOR_GRAY2RGB);
      *format = IMAGE_RGB8;
      break;
    case 3:
      cv::cvtColor(uchar_image, render_image, cv::COLOR_BGR2RGB);
      *format = IMAGE_RGB8;
      break;
    case 4:
      cv::cvtColor(uchar_image, render_image, cv::COLOR_BGRA2RGBA);
      *format = IMAGE_RGBA8;
      break;
    default:
      throw std::invalid_argument("only 1-, 3-, and 4-channel images can be shown");
  }
  return render_image.isContinuous() ? render_image : render_image.clone();
}

void render_cpu_image(const std::string& label, cv::Mat image, bool wait, bool squish) {
  imageFormat format = IMAGE_UNKNOWN;
  cv::Mat render_image = prepare_cpu_preview_image(std::move(image), squish, &format);
  void* device_image = nullptr;
  const std::size_t bytes = render_image.total() * render_image.elemSize();
  const cudaError_t alloc_status = cudaMalloc(&device_image, bytes);
  if (alloc_status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed for preview image: ") + cudaGetErrorString(alloc_status));
  }
  const cudaError_t copy_status = cudaMemcpy(device_image, render_image.data, bytes, cudaMemcpyHostToDevice);
  if (copy_status != cudaSuccess) {
    cudaFree(device_image);
    throw std::runtime_error(std::string("cudaMemcpy failed for preview image: ") + cudaGetErrorString(copy_status));
  }
  glDisplay* window = get_cpu_image_window(label, render_image.cols, render_image.rows);
  if (window) {
    window->Render(device_image, render_image.cols, render_image.rows, format);
    if (wait) {
      wait_for_preview_key(window);
    }
  }
  cudaFree(device_image);
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
  render_cpu_image(label, resize_for_preview(img, scale), wait, squish);
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
  const bool destroyed = gl_window || !cpu_image_windows.empty();
  if (gl_window) {
    gl_window.reset();
  }
  cpu_image_windows.clear();
  if (!destroyed) {
    return false;
  }
  return true;
}

void display_scaled_image(const std::string& label, cv::Mat image, float scale, bool wait, bool squish) {
  render_cpu_image(label, resize_for_preview(image, scale), wait, squish);
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
