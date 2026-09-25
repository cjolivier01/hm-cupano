#include <gtest/gtest.h>
#include <cstring>
#include "cupano/pano/cudaPano.h"
#include "cupano/pano/cudaPano3.h"
#include "cupano/pano/cudaPanoN.h"

namespace {
using namespace hm;
using namespace hm::pano;
constexpr int W = 769, H = 35, B = 2;

// A padded row and two contiguous images exercise pitch and batch addressing.
struct Inputs {
  std::vector<std::unique_ptr<CudaMat<Rgb10A2>>> storage, packed;
  std::vector<std::unique_ptr<CudaMat<half4>>> unpacked;
  std::vector<const CudaMat<Rgb10A2>*> packed_ptrs;
  std::vector<const CudaMat<half4>*> unpacked_ptrs;
  void fill(int n, int frame) {
    for (int i = 0; i < n; ++i) {
      std::vector<Rgb10A2> words((W + 7) * H * B);
      std::vector<half4> halves(W * H * B);
      for (int b = 0; b < B; ++b)
        for (int y = 0; y < H; ++y)
          for (int x = 0; x < W; ++x) {
            const unsigned k = (b * H + y) * W + x + i * 59 + frame * 101;
            const unsigned r = k % 1024, g = (k * 3) % 1024, blue = (k * 7) % 1024;
            words[(b * H + y) * (W + 7) + x].value = r | (g << 10) | (blue << 20) | ((k % 4) << 30);
            constexpr float scale = 255.0f / 1023.0f;
            halves[(b * H + y) * W + x] = {
                __float2half(r * scale), __float2half(g * scale), __float2half(blue * scale), __float2half(255.0f)};
          }
      storage.push_back(std::make_unique<CudaMat<Rgb10A2>>(B, W + 7, H));
      ASSERT_EQ(
          cudaSuccess,
          cudaMemcpy(storage.back()->data(), words.data(), words.size() * sizeof(Rgb10A2), cudaMemcpyHostToDevice));
      packed.push_back(std::make_unique<CudaMat<Rgb10A2>>(SurfaceInfo{W, H, (W + 7) * 4, storage.back()->data()}, B));
      unpacked.push_back(std::make_unique<CudaMat<half4>>(B, W, H));
      ASSERT_EQ(
          cudaSuccess,
          cudaMemcpy(unpacked.back()->data(), halves.data(), halves.size() * sizeof(half4), cudaMemcpyHostToDevice));
      packed_ptrs.push_back(packed.back().get());
      unpacked_ptrs.push_back(unpacked.back().get());
    }
  }
};

ControlMasksN masks_n(int n, bool invalid_maps = true) {
  ControlMasksN m;
  for (int i = 0; i < n; ++i) {
    cv::Mat mx(H, W, CV_16U), my(H, W, CV_16U);
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        mx.at<uint16_t>(y, x) = (x + y) % 31 == 0 ? 65535 : ((x + y) % 37 == 0 ? W + 5 : x);
        my.at<uint16_t>(y, x) = (x + y) % 41 == 0 ? 65535 : y;
      }
    if (!invalid_maps) {
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x) {
          mx.at<uint16_t>(y, x) = x;
          my.at<uint16_t>(y, x) = y;
        }
    }
    m.img_col.push_back(mx);
    m.img_row.push_back(my);
    m.positions.push_back(SpatialTiff{float(i * 24), float(i % 2 ? 3 : 0)});
  }
  m.whole_seam_mask_indexed = cv::Mat(H + 3, W + (n - 1) * 24, CV_8U);
  for (int x = 0; x < m.whole_seam_mask_indexed.cols; ++x)
    m.whole_seam_mask_indexed.col(x).setTo(std::min(n - 1, std::max(0, (x - 48) / 24)));
  if (!invalid_maps) {
    m.positions.assign(n, SpatialTiff{0.0f, 0.0f});
    m.whole_seam_mask_indexed = cv::Mat(H, W, CV_8U);
    for (int x = 0; x < W; ++x)
      m.whole_seam_mask_indexed.col(x).setTo(std::min(n - 1, x * n / W));
  }
  return m;
}

template <class Pano, class Run>
void parity(Pano& reference, Pano& fused, int n, Run run) {
  ASSERT_TRUE(reference.status().ok()) << reference.status().message();
  ASSERT_TRUE(fused.status().ok()) << fused.status().message();
  cudaStream_t stream;
  ASSERT_EQ(cudaSuccess, cudaStreamCreate(&stream));
  for (int frame = 0; frame < 3; ++frame) {
    Inputs inputs;
    inputs.fill(n, frame);
    auto old_result = run(reference, inputs.unpacked_ptrs, stream);
    auto new_result = run(fused, inputs.packed_ptrs, stream);
    ASSERT_TRUE(old_result.ok()) << old_result.status().message();
    ASSERT_TRUE(new_result.ok()) << new_result.status().message();
    ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));
    auto a = old_result.ConsumeValueOrDie(), b = new_result.ConsumeValueOrDie();
    ASSERT_EQ(a->size(), b->size());
    std::vector<unsigned char> ah(a->size()), bh(b->size());
    ASSERT_EQ(cudaSuccess, cudaMemcpy(ah.data(), a->data(), ah.size(), cudaMemcpyDeviceToHost));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(bh.data(), b->data(), bh.size(), cudaMemcpyDeviceToHost));
    EXPECT_EQ(ah, bh) << "frame=" << frame;
  }
  EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream));
}

TEST(Rgb10Remap, TwoCameras) {
  for (bool invalid_maps : {false, true}) {
    auto mn = masks_n(2, invalid_maps);
    ControlMasks m;
    m.img1_col = mn.img_col[0];
    m.img1_row = mn.img_row[0];
    m.img2_col = mn.img_col[1];
    m.img2_row = mn.img_row[1];
    m.positions = mn.positions;
    m.whole_seam_mask_image = 1 - mn.whole_seam_mask_indexed;
    for (int levels : {0, 1, 4})
      for (bool minimize : {false, true}) {
        SCOPED_TRACE(::testing::Message() << levels << '/' << minimize);
        cuda::CudaStitchPano<half4, half4> a(B, levels, m, true, minimize, 0), b(B, levels, m, true, minimize, 0);
        if (minimize && levels > 0 && !invalid_maps) {
          ASSERT_TRUE(a.minimizes_blend());
          ASSERT_TRUE(b.minimizes_blend());
        }
        parity(a, b, 2, [](auto& pano, const auto& inputs, auto stream) {
          auto canvas = std::make_unique<CudaMat<half4>>(B, pano.canvas_width(), pano.canvas_height());
          return pano.process(*inputs[0], *inputs[1], stream, std::move(canvas));
        });
      }
  }
}
TEST(Rgb10Remap, ThreeCameras) {
  for (bool invalid_maps : {false, true}) {
    auto mn = masks_n(3, invalid_maps);
    ControlMasks3 m;
    m.img0_col = mn.img_col[0];
    m.img0_row = mn.img_row[0];
    m.img1_col = mn.img_col[1];
    m.img1_row = mn.img_row[1];
    m.img2_col = mn.img_col[2];
    m.img2_row = mn.img_row[2];
    m.positions = mn.positions;
    m.whole_seam_mask_image = mn.whole_seam_mask_indexed;
    for (bool fused : {false, true})
      for (int levels : {0, 1, 4})
        for (bool minimize : {false, true}) {
          SCOPED_TRACE(::testing::Message() << levels << '/' << minimize);
          cuda::CudaStitchPano3<half4, half4> a(B, levels, m, true, 0, minimize), b(B, levels, m, true, 0, minimize);
          if (minimize && levels > 0 && !invalid_maps) {
            ASSERT_TRUE(a.minimizes_blend());
            ASSERT_TRUE(b.minimizes_blend());
          }
          parity(a, b, 3, [fused](auto& pano, const auto& inputs, auto stream) {
            auto canvas = std::make_unique<CudaMat<half4>>(B, pano.canvas_width(), pano.canvas_height());
            return pano.process(*inputs[0], *inputs[1], *inputs[2], stream, std::move(canvas), fused);
          });
        }
  }
}
TEST(Rgb10Remap, NCameras) {
  for (bool invalid_maps : {false, true}) {
    for (int n : {2, 3, 4, 8}) {
      auto m = masks_n(n, invalid_maps);
      for (int levels : {0, 1, 4})
        for (bool minimize : {false, true}) {
          SCOPED_TRACE(::testing::Message() << n << '/' << levels << '/' << minimize);
          cuda::CudaStitchPanoN<half4, half4> a(B, levels, m, minimize, true, 0), b(B, levels, m, minimize, true, 0);
          if (minimize && levels > 0 && !invalid_maps && n <= 4) {
            ASSERT_TRUE(a.minimizes_blend());
            ASSERT_TRUE(b.minimizes_blend());
          }
          parity(a, b, n, [](auto& pano, const auto& inputs, auto stream) {
            auto canvas = std::make_unique<CudaMat<half4>>(B, pano.canvas_width(), pano.canvas_height());
            return pano.process(inputs, stream, std::move(canvas));
          });
        }
    }
  }
}
} // namespace
