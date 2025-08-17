#include "cudaPano3.h"
#include <array>
#include <optional>
#include <cmath>

namespace hm {
namespace pano {

// Helper: clamp float image back to valid range if needed
static inline void adjust_image_add(cv::Mat& img, const cv::Scalar& add) {
  if (img.empty()) return;
  if (img.depth() == CV_8U) {
    cv::Mat f;
    img.convertTo(f, CV_32F);
    std::vector<cv::Mat> ch;
    cv::split(f, ch);
    for (int k = 0; k < 3; ++k) ch[k] += static_cast<float>(add[k]);
    cv::merge(ch, f);
    cv::min(f, 255.0, f);
    cv::max(f, 0.0, f);
    f.convertTo(img, CV_8U);
  } else {
    std::vector<cv::Mat> ch;
    cv::split(img, ch);
    for (int k = 0; k < 3; ++k) ch[k] += static_cast<float>(add[k]);
    cv::merge(ch, img);
  }
}

// Compute per-image additive RGB offsets to minimize seams across all overlapping pairs
std::optional<std::array<cv::Scalar, 3>> match_seam_images3(
    const cv::Mat& image0,
    const cv::Mat& image1,
    const cv::Mat& image2,
    const cv::Mat& seam_indexed, // CV_8U with labels {0,1,2}
    int N,
    const cv::Point& topLeft0,
    const cv::Point& topLeft1,
    const cv::Point& topLeft2,
    bool verbose) {
  if (seam_indexed.type() != CV_8U) {
    std::cerr << "Error: 3-image seam mask must be CV_8U indexed." << std::endl;
    return std::nullopt;
  }
  if (image0.empty() || image1.empty() || image2.empty()) return std::nullopt;
  if (image0.channels() < 3 || image1.channels() < 3 || image2.channels() < 3) return std::nullopt;

  struct Accum {
    cv::Scalar sumA{0, 0, 0};
    cv::Scalar sumB{0, 0, 0};
    int cntA{0};
    int cntB{0};
  };
  // Pairs stored with (a<b): 01, 02, 12
  Accum pair01, pair02, pair12;

  auto add_pixel = [](const cv::Mat& img, int r, int c, cv::Scalar& acc) {
    if (img.depth() == CV_8U) {
      const cv::Vec3b pix = img.at<cv::Vec3b>(r, c);
      acc[0] += pix[0];
      acc[1] += pix[1];
      acc[2] += pix[2];
    } else { // float32
      const cv::Vec3f pix = img.at<cv::Vec3f>(r, c);
      acc[0] += pix[0];
      acc[1] += pix[1];
      acc[2] += pix[2];
    }
  };

  auto in_bounds = [](const cv::Mat& img, int r, int c) {
    return r >= 0 && r < img.rows && c >= 0 && c < img.cols;
  };

  // For each row in seam, scan for label changes; collect N-pixel strips on each side
  for (int gr = 0; gr < seam_indexed.rows; ++gr) {
    const uchar* row = seam_indexed.ptr<uchar>(gr);
    int prev_label = row[0];
    for (int gc = 1; gc < seam_indexed.cols; ++gc) {
      int curr_label = row[gc];
      if (curr_label == prev_label) continue;
      // Boundary between prev_label and curr_label at column gc
      int a = std::min(prev_label, curr_label);
      int b = std::max(prev_label, curr_label);

      // Sample N pixels from side of 'prev_label' just before boundary
      auto sample_side = [&](int label, int gcol_start, int gcol_end, cv::Scalar& sum, int& cnt) {
        const cv::Point tl = (label == 0 ? topLeft0 : (label == 1 ? topLeft1 : topLeft2));
        const cv::Mat& src = (label == 0 ? image0 : (label == 1 ? image1 : image2));
        for (int gc2 = gcol_start; gc2 < gcol_end; ++gc2) {
          int lr = gr - tl.y;
          int lc = gc2 - tl.x;
          if (!in_bounds(src, lr, lc)) continue;
          add_pixel(src, lr, lc, sum);
          cnt++;
        }
      };

      // define strips [gc-N, gc) belongs to prev_label; [gc, gc+N) belongs to curr_label
      int left_start = std::max(0, gc - N);
      int left_end = gc;
      int right_start = gc;
      int right_end = std::min(seam_indexed.cols, gc + N);

      if (a == 0 && b == 1) {
        if (prev_label == 0) {
          sample_side(0, left_start, left_end, pair01.sumA, pair01.cntA);
          sample_side(1, right_start, right_end, pair01.sumB, pair01.cntB);
        } else {
          sample_side(1, left_start, left_end, pair01.sumB, pair01.cntB);
          sample_side(0, right_start, right_end, pair01.sumA, pair01.cntA);
        }
      } else if (a == 0 && b == 2) {
        if (prev_label == 0) {
          sample_side(0, left_start, left_end, pair02.sumA, pair02.cntA);
          sample_side(2, right_start, right_end, pair02.sumB, pair02.cntB);
        } else {
          sample_side(2, left_start, left_end, pair02.sumB, pair02.cntB);
          sample_side(0, right_start, right_end, pair02.sumA, pair02.cntA);
        }
      } else if (a == 1 && b == 2) {
        if (prev_label == 1) {
          sample_side(1, left_start, left_end, pair12.sumA, pair12.cntA);
          sample_side(2, right_start, right_end, pair12.sumB, pair12.cntB);
        } else {
          sample_side(2, left_start, left_end, pair12.sumB, pair12.cntB);
          sample_side(1, right_start, right_end, pair12.sumA, pair12.cntA);
        }
      }

      prev_label = curr_label;
    }
  }

  // Build Laplacian L and rhs b for each channel
  // Minimize sum_w (a_i - a_j + d_ij)^2 where d_ij = mean_i - mean_j
  double L[3][3] = {{0}};
  cv::Vec3d bvec[3] = {cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0), cv::Vec3d(0, 0, 0)};

  auto accumulate_pair = [&](int i, int j, const Accum& P) {
    if (P.cntA == 0 || P.cntB == 0) return;
    cv::Scalar avgA = P.sumA * (1.0 / P.cntA);
    cv::Scalar avgB = P.sumB * (1.0 / P.cntB);
    cv::Vec3d d(avgA[0] - avgB[0], avgA[1] - avgB[1], avgA[2] - avgB[2]);
    double w = static_cast<double>(std::min(P.cntA, P.cntB));
    // L updates
    L[i][i] += w;
    L[j][j] += w;
    L[i][j] -= w;
    L[j][i] -= w;
    // b updates: a_i - a_j = -d  => b_i += -w*d; b_j += +w*d
    bvec[i] += (-w) * d;
    bvec[j] += (+w) * d;
  };

  accumulate_pair(0, 1, pair01);
  accumulate_pair(0, 2, pair02);
  accumulate_pair(1, 2, pair12);

  // Solve with gauge a0=0; solve 2x2 for a1,a2 for each channel
  auto solve_sub = [&](int c) -> std::array<double, 3> {
    double A11 = L[1][1], A12 = L[1][2];
    double A21 = L[2][1], A22 = L[2][2];
    double r1 = bvec[1][c];
    double r2 = bvec[2][c];
    double det = A11 * A22 - A12 * A21;
    double a0 = 0.0, a1 = 0.0, a2 = 0.0;
    if (std::fabs(det) > 1e-9) {
      a1 = (r1 * A22 - A12 * r2) / det;
      a2 = (A11 * r2 - r1 * A21) / det;
    } else {
      a1 = 0.0;
      a2 = 0.0;
    }
    return {a0, a1, a2};
  };

  std::array<cv::Scalar, 3> adj;
  auto solB = solve_sub(0);
  auto solG = solve_sub(1);
  auto solR = solve_sub(2);
  for (int i = 0; i < 3; ++i) {
    adj[i] = cv::Scalar(solB[i], solG[i], solR[i]);
  }

  if (verbose) {
    std::cout << "3-image seam adjustments:"
              << " a0=" << adj[0] << " a1=" << adj[1] << " a2=" << adj[2] << std::endl;
  }
  return adj;
}

} // namespace pano
} // namespace hm
