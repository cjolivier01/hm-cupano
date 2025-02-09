#include "showImage.h"

#include <set>
#include <unordered_set>

#include <opencv2/opencv.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <fcntl.h>
#include <stdio.h>
#include <termios.h>
#include <unistd.h>

namespace hm {
namespace utils {
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

int wait_key() {
  int c;
  while (!(c = kbhit())) {
    usleep(100);
  }
  return c;
}

void show_image(const std::string& label, const cv::Mat& img, bool wait) {
  cv::imshow(label, img);
  cv::waitKey(wait ? 0 : 1);
}

void display_scaled_image(const std::string& label, cv::Mat image, float scale, bool wait) {
  if (scale != 1.0f) {
    // Calculate new dimensions
    int newWidth = static_cast<int>(image.cols * scale);
    int newHeight = static_cast<int>(image.rows * scale);

    // Resize the image
    cv::resize(image, image, cv::Size(newWidth, newHeight));
  }

  // Display the image
  cv::imshow(label, image);
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

// cv::Mat load_position_mask(const std::string& filename, double* minVal, double* maxVal) {
//   cv::Mat pos_mask = cv::imread(filename, cv::IMREAD_ANYDEPTH);
//   if (!pos_mask.empty()) {
//     if (minVal || maxVal) {
//       cv::Point minLoc, maxLoc;
//       // Get the minimum and maximum values and their locations
//       double min, max;
//       cv::minMaxLoc(pos_mask, &min, &max, &minLoc, &maxLoc);
//       if (minVal) {
//         *minVal = min;
//       }
//       if (maxVal) {
//         *maxVal = max;
//       }
//     }
//   } else {
//     if (minVal) {
//       *minVal = std::nan("");
//     }
//     if (maxVal) {
//       *maxVal = std::nan("");
//     }
//   }
//   return pos_mask;
// }

cv::Mat make_fake_mask_like(const cv::Mat& mask) {
  cv::Mat img(mask.rows, mask.cols, CV_32FC1, cv::Scalar(0));

  // Define a region of interest (ROI) for the left half of the image.
  cv::Rect leftHalfROI(0, 0, mask.cols / 2, mask.rows);

  // Set all pixels in the left half to 1.
  img(leftHalfROI).setTo(1.0f);
  return img;
}

} // namespace utils
} // namespace hm
