/* clang-format off */
// cudaBlendN_test.cpp
//
// Google Test for N-way Laplacian blending (templates in cudaBlendN.h/cudaBlendN.cu).
/* clang-format on */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cudaBlendN.h"

#include <vector>

#define CUDA_CHECK(call)                                                                                      \
  do {                                                                                                        \
    cudaError_t err = (call);                                                                                 \
    if (err != cudaSuccess) {                                                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));              \
      FAIL() << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << static_cast<int>(err) << " \"" \
             << cudaGetErrorString(err) << "\"";                                                              \
    }                                                                                                         \
  } while (0)

constexpr float kTol = 1e-5f;

// Test: Single-pixel RGBA, alpha==0 image should be ignored in N-way blend
TEST(CudaBlendNSmallTest, AlphaZeroSkipsContribution) {
  constexpr int W = 1, H = 1, C = 4, B = 1, L = 1; // single-level, no pyramids

  // Three inputs, but one is fully transparent
  std::vector<float> h_img1{10, 20, 30, 255};
  std::vector<float> h_img2{100, 110, 120, 0};
  std::vector<float> h_img3{90, 100, 110, 255};
  float* d_img1 = nullptr;
  float* d_img2 = nullptr;
  float* d_img3 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img1, h_img1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_img2, h_img2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_img3, h_img3.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_img1, h_img1.data(), h_img1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_img2, h_img2.data(), h_img2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_img3, h_img3.data(), h_img3.size() * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<const float*> d_imgs{d_img1, d_img2, d_img3};

  // Mask: mostly selects img2, but alpha==0 → should renormalize to imgs 1 and 3 only
  std::vector<float> h_mask{0.1f, 0.8f, 0.1f};
  std::vector<float> h_out(W * H * C * B, 0.0f);
  // Device buffers for mask/output
  float* d_mask = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_out.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, h_out.size() * sizeof(float)));

  CudaBatchLaplacianBlendContextN<float, 3> ctx(W, H, L, B);
  ASSERT_EQ((cudaBatchedLaplacianBlendWithContextN<float, float, 3, 4>(d_imgs, d_mask, d_out, ctx, 0)), cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Expected: weights -> 0.5, 0.0, 0.5
  std::vector<float> expected{(10.0f + 90.0f) * 0.5f, (20.0f + 100.0f) * 0.5f, (30.0f + 110.0f) * 0.5f, 255.0f};
  for (int c = 0; c < C; ++c) {
    EXPECT_NEAR(h_out[c], expected[c], kTol) << "Mismatch at channel " << c;
  }
  cudaFree(d_img1);
  cudaFree(d_img2);
  cudaFree(d_img3);
  cudaFree(d_mask);
  cudaFree(d_out);
}

// Test: Multi-level blend of constant inputs should match per-pixel weighted average.
TEST(CudaBlendNTest, MultiLevelConstantBlendMatchesWeights) {
  constexpr int W = 8, H = 8, C = 4, B = 2, L = 4;
  constexpr int N = 2;

  const float w0 = 0.25f;
  const float w1 = 0.75f;

  const int pixels = W * H * B;
  std::vector<float> h_img0(pixels * C);
  std::vector<float> h_img1(pixels * C);
  for (int p = 0; p < pixels; ++p) {
    int base = p * C;
    h_img0[base + 0] = 10.0f;
    h_img0[base + 1] = 20.0f;
    h_img0[base + 2] = 30.0f;
    h_img0[base + 3] = 255.0f;
    h_img1[base + 0] = 110.0f;
    h_img1[base + 1] = 120.0f;
    h_img1[base + 2] = 130.0f;
    h_img1[base + 3] = 255.0f;
  }

  std::vector<float> h_mask(W * H * N);
  for (int p = 0; p < W * H; ++p) {
    h_mask[p * N + 0] = w0;
    h_mask[p * N + 1] = w1;
  }

  float* d_img0 = nullptr;
  float* d_img1 = nullptr;
  float* d_mask = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img0, h_img0.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_img1, h_img1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, h_img0.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_img0, h_img0.data(), h_img0.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_img1, h_img1.data(), h_img1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, h_img0.size() * sizeof(float)));

  std::vector<const float*> d_imgs{d_img0, d_img1};
  CudaBatchLaplacianBlendContextN<float, N> ctx(W, H, L, B);
  ASSERT_EQ((cudaBatchedLaplacianBlendWithContextN<float, float, N, C>(d_imgs, d_mask, d_out, ctx, 0)), cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(h_img0.size(), 0.0f);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  const float exp_r = w0 * 10.0f + w1 * 110.0f;
  const float exp_g = w0 * 20.0f + w1 * 120.0f;
  const float exp_b = w0 * 30.0f + w1 * 130.0f;
  const float exp_a = 255.0f;
  for (int p = 0; p < pixels; ++p) {
    int base = p * C;
    EXPECT_NEAR(h_out[base + 0], exp_r, kTol) << "Mismatch at pixel " << p << " channel 0";
    EXPECT_NEAR(h_out[base + 1], exp_g, kTol) << "Mismatch at pixel " << p << " channel 1";
    EXPECT_NEAR(h_out[base + 2], exp_b, kTol) << "Mismatch at pixel " << p << " channel 2";
    EXPECT_NEAR(h_out[base + 3], exp_a, kTol) << "Mismatch at pixel " << p << " channel 3";
  }

  cudaFree(d_img0);
  cudaFree(d_img1);
  cudaFree(d_mask);
  cudaFree(d_out);
}

// Test: If mask selects only a transparent source (alpha==0), blend should fall back to a valid contributor.
TEST(CudaBlendNTest, MaskSelectsTransparentThenFallback) {
  constexpr int W = 1, H = 1, C = 4, B = 1, L = 1;
  constexpr int N = 3;

  // image1: solid red, alpha 255
  std::vector<float> h_img1{255.0f, 0.0f, 0.0f, 255.0f};
  // image2: green, but fully transparent
  std::vector<float> h_img2{0.0f, 255.0f, 0.0f, 0.0f};
  // image3: blue, lower alpha
  std::vector<float> h_img3{0.0f, 0.0f, 255.0f, 128.0f};
  std::vector<const float*> d_imgs;

  float* d_img1 = nullptr;
  float* d_img2 = nullptr;
  float* d_img3 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_img1, h_img1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_img2, h_img2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_img3, h_img3.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_img1, h_img1.data(), h_img1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_img2, h_img2.data(), h_img2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_img3, h_img3.data(), h_img3.size() * sizeof(float), cudaMemcpyHostToDevice));
  d_imgs = {d_img1, d_img2, d_img3};

  // Mask selects only image2
  std::vector<float> h_mask{0.0f, 1.0f, 0.0f};
  float* d_mask = nullptr;
  float* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_out, W * H * C * B * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_out, 0, W * H * C * B * sizeof(float)));

  CudaBatchLaplacianBlendContextN<float, N> ctx(W, H, L, B);
  ASSERT_EQ((cudaBatchedLaplacianBlendWithContextN<float, float, N, C>(d_imgs, d_mask, d_out, ctx, 0)), cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<float> h_out(W * H * C * B, 0.0f);
  CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), cudaMemcpyDeviceToHost));

  // Expect fallback picked image1 (highest alpha 255).
  std::vector<float> expected{255.0f, 0.0f, 0.0f, 255.0f};
  for (int c = 0; c < C; ++c) {
    EXPECT_NEAR(h_out[c], expected[c], kTol) << "Channel " << c << " mismatch in fallback.";
  }

  cudaFree(d_img1);
  cudaFree(d_img2);
  cudaFree(d_img3);
  cudaFree(d_mask);
  cudaFree(d_out);
}
