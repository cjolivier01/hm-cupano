/* clang-format off */
// cudaBlendN_test.cpp
//
// Google Test for N-way Laplacian blending (templates in cudaBlendN.h/cudaBlendN.cu).
/* clang-format on */

#include <cupano/gpu/gpu_runtime.h>
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
  std::vector<float> img1{10, 20, 30, 255};
  std::vector<float> img2{100, 110, 120, 0};
  std::vector<float> img3{90, 100, 110, 255};
  std::vector<const float*> d_imgs{img1.data(), img2.data(), img3.data()};

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
  std::vector<float> expected{(10.0f + 90.0f) * 0.5f,
                              (20.0f + 100.0f) * 0.5f,
                              (30.0f + 110.0f) * 0.5f,
                              255.0f};
  for (int c = 0; c < C; ++c) {
    EXPECT_NEAR(h_out[c], expected[c], kTol) << "Mismatch at channel " << c;
  }
  cudaFree(d_mask);
  cudaFree(d_out);
}
