// Dumps the C++ feather field for a named fixture so scripts/compare_feather_parity.py can check
// it against cupano.feather.build_weights element by element. The parity claim in AGENTS.md and
// docs/alpha-blend-mode.md rests on this comparison, so the harness lives in the tree rather than
// being rebuilt by hand each time someone wants to re-run it.
//
//   bazel run //src/pano:featherMaskDump -- two_camera /tmp/two_camera.bin
//   python3 scripts/compare_feather_parity.py two_camera /tmp/two_camera.bin

#include <opencv2/core.hpp>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "cupano/pano/featherMask.h"

namespace {

constexpr uint16_t kUnmapped = 65535;

struct Fixture {
  cv::Mat labels;
  std::vector<cv::Mat> remap_x;
  std::vector<cv::Mat> remap_y;
  std::vector<cv::Point> positions;
  float fraction{0.1f};
};

cv::Mat mapped(int w, int h) {
  return cv::Mat(h, w, CV_16U, cv::Scalar(0));
}

cv::Mat unmapped(int w, int h) {
  return cv::Mat(h, w, CV_16U, cv::Scalar(kUnmapped));
}

// Keep these in lockstep with scripts/compare_feather_parity.py; the two build the same rigs.
bool build(const std::string& name, Fixture* out) {
  if (name == "two_camera") {
    out->labels = cv::Mat(200, 600, CV_8U, cv::Scalar(0));
    out->labels.colRange(250, 600).setTo(1);
    out->remap_x = {mapped(400, 200), mapped(450, 200)};
    out->remap_y = {mapped(400, 200), mapped(450, 200)};
    out->positions = {{0, 0}, {150, 0}};
    out->fraction = 0.5f;
    return true;
  }
  // Camera 2 covers the canvas but owns no label, so its weight is zero everywhere. Kept because
  // it is the only rig here with an empty label region, which is what carries the saturated
  // distance transform across the language boundary.
  if (name == "empty_label_region") {
    out->labels = cv::Mat(200, 600, CV_8U, cv::Scalar(0));
    out->labels.colRange(250, 600).setTo(1);
    out->remap_x = {mapped(400, 200), mapped(450, 200), mapped(600, 200)};
    out->remap_y = {mapped(400, 200), mapped(450, 200), mapped(600, 200)};
    out->positions = {{0, 0}, {150, 0}, {0, 0}};
    out->fraction = 0.5f;
    return true;
  }
  // Camera 2 covers the canvas and owns a sliver, so it does reach the cap: the case where a
  // count-based "covered by at least two" test would stop the cap engaging.
  if (name == "blanketing_third") {
    out->labels = cv::Mat(200, 600, CV_8U, cv::Scalar(0));
    out->labels.colRange(250, 600).setTo(1);
    out->labels.colRange(290, 310).setTo(2);
    out->remap_x = {mapped(400, 200), mapped(450, 200), mapped(600, 200)};
    out->remap_y = {mapped(400, 200), mapped(450, 200), mapped(600, 200)};
    out->positions = {{0, 0}, {150, 0}, {0, 0}};
    out->fraction = 0.5f;
    return true;
  }
  if (name == "grazing_neighbour") {
    out->labels = cv::Mat(400, 600, CV_8U, cv::Scalar(0));
    out->labels.colRange(250, 600).setTo(1);
    out->labels(cv::Rect(310, 198, 20, 5)).setTo(2);
    cv::Mat sliver = unmapped(400, 400);
    sliver(cv::Rect(0, 198, 134, 5)).setTo(0);
    out->remap_x = {mapped(400, 400), mapped(400, 400), sliver};
    out->remap_y = {mapped(400, 400), mapped(400, 400), sliver.clone()};
    out->positions = {{0, 0}, {150, 0}, {198, 0}};
    out->fraction = 1.0f;
    return true;
  }
  if (name == "coverage_hole") {
    out->labels = cv::Mat(300, 500, CV_8U, cv::Scalar(0));
    out->labels.colRange(250, 500).setTo(1);
    cv::Mat holed = mapped(500, 300);
    holed(cv::Rect(120, 60, 90, 90)).setTo(kUnmapped);
    out->remap_x = {holed, mapped(500, 300)};
    out->remap_y = {holed.clone(), mapped(500, 300)};
    out->positions = {{0, 0}, {0, 0}};
    out->fraction = 0.2f;
    return true;
  }
  if (name == "hard_fallback") {
    out->labels = cv::Mat(40, 128, CV_8U, cv::Scalar(0));
    out->labels.colRange(60, 128).setTo(1);
    cv::Mat holed = mapped(128, 40);
    holed(cv::Rect(20, 8, 30, 20)).setTo(kUnmapped);
    out->remap_x = {holed, mapped(128, 40)};
    out->remap_y = {holed.clone(), mapped(128, 40)};
    out->positions = {{0, 0}, {0, 0}};
    out->fraction = 0.0f;
    return true;
  }
  if (name == "eight_camera") {
    constexpr int w = 300, h = 200, n = 8, stride = 150;
    const int canvas_w = w + stride * (n - 1);
    out->labels = cv::Mat(h, canvas_w, CV_8U);
    for (int x = 0; x < canvas_w; ++x) {
      out->labels.col(x).setTo(std::min(n - 1, x * n / canvas_w));
    }
    for (int i = 0; i < n; ++i) {
      out->remap_x.push_back(mapped(w, h));
      out->remap_y.push_back(mapped(w, h));
      out->positions.emplace_back(stride * i, 0);
    }
    out->fraction = 0.15f;
    return true;
  }
  return false;
}

} // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: featherMaskDump <fixture> <output.bin>\n"
              << "fixtures: two_camera empty_label_region blanketing_third grazing_neighbour "
                 "coverage_hole hard_fallback eight_camera\n";
    return 2;
  }
  Fixture fixture;
  if (!build(argv[1], &fixture)) {
    std::cerr << "unknown fixture: " << argv[1] << '\n';
    return 2;
  }

  hm::pano::feather::Params params;
  params.fraction = fixture.fraction;
  const hm::pano::feather::Result result = hm::pano::feather::build_weights(
      fixture.labels,
      fixture.remap_x,
      fixture.remap_y,
      fixture.positions,
      static_cast<int>(fixture.remap_x.size()),
      params);
  if (!result.error.empty()) {
    std::cerr << "build_weights failed: " << result.error << '\n';
    return 1;
  }

  std::FILE* out = std::fopen(argv[2], "wb");
  if (out == nullptr) {
    std::cerr << "cannot open " << argv[2] << '\n';
    return 1;
  }
  const int32_t header[3] = {result.weights.rows, result.weights.cols, result.weights.channels()};
  std::fwrite(header, sizeof(int32_t), 3, out);
  std::fwrite(result.weights.data, sizeof(float), static_cast<size_t>(header[0]) * header[1] * header[2], out);
  std::fwrite(result.corrected_labels.data, 1, static_cast<size_t>(header[0]) * header[1], out);
  const float metadata[5] = {
      result.radius_px,
      result.min_seam_radius_px,
      result.requested_radius_px,
      result.capped_seam_fraction,
      static_cast<float>(result.overlap_capped)};
  std::fwrite(metadata, sizeof(float), 5, out);
  std::fclose(out);
  return 0;
}
