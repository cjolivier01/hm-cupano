/* clang-format off */
// cudaBlend3_test.cpp
//
// Google Test suite for cudaBlend3.h / cudaBlend3.cu
// Verifies the correctness of the three-image Laplacian blending for simple scenarios.
//
// To compile (assuming CUDA, GTest, and your project are set up):
// nvcc -std=c++14 -I<path-to-gtest>/include -L<path-to-gtest>/lib64 cudaBlend3.cu cudaBlend3_test.cpp -lgtest -lgtest_main -lcudart -o cudaBlend3_test
//
// Then run:
// ./cudaBlend3_test
/* clang-format on */

#include <cupano/gpu/gpu_runtime.h>
#include <gtest/gtest.h>
#include "cudaBlend3.h"

#include <cassert>
#include <cmath>
#include <vector>

// Helper macro to check CUDA calls
#define CUDA_CHECK(call)                                                                                      \
  do {                                                                                                        \
    cudaError_t err = (call);                                                                                 \
    if (err != cudaSuccess) {                                                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));              \
      FAIL() << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << static_cast<int>(err) << " \"" \
             << cudaGetErrorString(err) << "\"";                                                              \
    }                                                                                                         \
  } while (0)

// Numeric tolerance for floating-point comparisons
constexpr float kEpsilon = 1e-5f;

struct CudaVector {
  using Type = float;
  CudaVector(std::vector<Type>& vec) : vec_(vec) {
    cudaError_t cuerr = cudaMalloc(&d_ptr, vec_.size() * sizeof(Type));
    (void)cuerr;
    assert(cuerr == cudaError_t::cudaSuccess);
    cuerr = cudaMemcpy(d_ptr, vec_.data(), vec_.size() * sizeof(Type), cudaMemcpyKind::cudaMemcpyHostToDevice);
    (void)cuerr;
    assert(cuerr == cudaError_t::cudaSuccess);
  }
  ~CudaVector() {
    if (d_ptr) {
      cudaError_t cuerr =
          cudaMemcpy(vec_.data(), d_ptr, vec_.size() * sizeof(Type), cudaMemcpyKind::cudaMemcpyDeviceToHost);
      (void)cuerr;
      assert(cuerr == cudaError_t::cudaSuccess);
      cudaFree(d_ptr);
    }
  }
  float* data() {
    return d_ptr;
  }
  float* d_ptr{nullptr};
  std::vector<float>& vec_;
};

// Test 1: Single-Pixel (1×1) RGB images, identity masks
// Ensure that when mask = (1,0,0), output = image1;
// when mask = (0,1,0), output = image2;
// when mask = (0,0,1), output = image3.
TEST(CudaBlend3SmallTest, SinglePixelIdentityMasks) {
  const int width = 1;
  const int height = 1;
  const int channels = 3; // RGB
  const int batchSize = 1;
  const int numLevels = 1; // No pyramid levels beyond the base
  const int pixelCount = width * height * channels * batchSize;
  const int maskCount = width * height * 3; // 3-channel mask per pixel

  // Allocate host arrays
  std::vector<float> h_image1(pixelCount);
  std::vector<float> h_image2(pixelCount);
  std::vector<float> h_image3(pixelCount);
  std::vector<float> h_mask(maskCount);
  std::vector<float> h_output(pixelCount);

  // Initialize images with distinct values
  // image1 = {10, 20, 30}, image2 = {40, 50, 60}, image3 = {70, 80, 90}
  h_image1 = {10.0f, 20.0f, 30.0f};
  h_image2 = {40.0f, 50.0f, 60.0f};
  h_image3 = {70.0f, 80.0f, 90.0f};

  CudaVector img1(h_image1), img2(h_image2), img3(h_image3);
  {
    // Case A: Mask = (1, 0, 0)
    h_mask = {1.0f, 0.0f, 0.0f};
    CudaVector mask(h_mask);
    CudaVector output(h_output);
    CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
    CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
    ASSERT_EQ(
        cudaBatchedLaplacianBlendWithContext3<float>(
            img1.data(),
            img2.data(),
            img3.data(),
            mask.data(),
            output.data(),
            ctx,
            channels,
            0),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Output should match image1 exactly
  for (int c = 0; c < channels; c++) {
    EXPECT_NEAR(h_output[c], h_image1[c], kEpsilon) << "Channel " << c << " mismatch for mask=(1,0,0)";
  }

  // Case B: Mask = (0, 1, 0)
  {
    h_mask = {0.0f, 1.0f, 0.0f};
    CudaVector mask(h_mask);
    CudaVector output(h_output);
    CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
    CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
    ASSERT_EQ(
        cudaBatchedLaplacianBlendWithContext3<float>(
            img1.data(),
            img2.data(),
            img3.data(),
            mask.data(),
            output.data(),
            ctx,
            channels,
            0),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Output should match image2
  for (int c = 0; c < channels; c++) {
    EXPECT_NEAR(h_output[c], h_image2[c], kEpsilon) << "Channel " << c << " mismatch for mask=(0,1,0)";
  }
  {
    // Case C: Mask = (0, 0, 1)
    h_mask = {0.0f, 0.0f, 1.0f};
    CudaVector mask(h_mask);
    CudaVector output(h_output);
    CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
    ASSERT_EQ(
        cudaBatchedLaplacianBlend3<float>(
            img1.data(),
            img2.data(),
            img3.data(),
            mask.data(),
            output.data(),
            width,
            height,
            channels,
            numLevels,
            batchSize,
            0),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Output should match image3
  for (int c = 0; c < channels; c++) {
    EXPECT_NEAR(h_output[c], h_image3[c], kEpsilon) << "Channel " << c << " mismatch for mask=(0,0,1)";
  }
}

// Test 2: 2×2 RGB images, uniform equal-weight mask
// image1 all ones, image2 all twos, image3 all threes, mask = (1/3, 1/3, 1/3)
// Then output pixel = (1 + 2 + 3) / 3 = 2 for each channel.
TEST(CudaBlend3SmallTest, TwoByTwoUniformMask) {
  const int width = 2;
  const int height = 2;
  const int channels = 3; // RGB
  const int batchSize = 1;
  const int numLevels = 1; // Single level (no downsampling)
  const int pixelCount = width * height * channels * batchSize;
  const int maskCount = width * height * 3; // 3-channel mask per pixel

  // Allocate host arrays
  std::vector<float> h_image1(pixelCount, 1.0f); // all ones
  std::vector<float> h_image2(pixelCount, 2.0f); // all twos
  std::vector<float> h_image3(pixelCount, 3.0f); // all threes
  std::vector<float> h_mask(maskCount);
  std::vector<float> h_output(pixelCount);

  CudaVector img1(h_image1), img2(h_image2), img3(h_image3);

  // Uniform mask = (1/3, 1/3, 1/3) at each pixel
  for (int i = 0; i < width * height; i++) {
    h_mask[3 * i + 0] = 1.0f / 3.0f;
    h_mask[3 * i + 1] = 1.0f / 3.0f;
    h_mask[3 * i + 2] = 1.0f / 3.0f;
  }
  CudaVector mask(h_mask);
  {
    CudaVector output(h_output);
    CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
    CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
    ASSERT_EQ(
        cudaBatchedLaplacianBlendWithContext3<float>(
            img1.data(),
            img2.data(),
            img3.data(),
            mask.data(),
            output.data(),
            ctx,
            channels,
            0),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  // Expect each blended pixel channel to be exactly 2.0
  for (int idx = 0; idx < pixelCount; idx++) {
    EXPECT_NEAR(h_output[idx], 2.0f, kEpsilon) << "Output mismatch at index " << idx;
  }
}

// Test 3: Single-Pixel RGBA images (channels=4), varying alpha blending
// Verify that RGB gets blended by (m₁,m₂,m₃) but alpha is also weighted by same mask.
TEST(CudaBlend3SmallTest, SinglePixelRGBAAlphaBlending) {
  const int width = 1;
  const int height = 1;
  const int channels = 4; // RGBA
  const int batchSize = 1;
  const int numLevels = 1;
  const int pixelCount = width * height * channels * batchSize;
  const int maskCount = width * height * 3; // mask has 3 weights

  std::vector<float> h_image1(pixelCount);
  std::vector<float> h_image2(pixelCount);
  std::vector<float> h_image3(pixelCount);
  std::vector<float> h_mask(maskCount);
  std::vector<float> h_output(pixelCount);

  // image1 = [R=10, G=20, B=30, A=40]
  // image2 = [R=50, G=60, B=70, A=80]
  // image3 = [R=90, G=100, B=110, A=120]
  h_image1 = {10.0f, 20.0f, 30.0f, 40.0f};
  h_image2 = {50.0f, 60.0f, 70.0f, 80.0f};
  h_image3 = {90.0f, 100.0f, 110.0f, 120.0f};

  CudaVector img1(h_image1), img2(h_image2), img3(h_image3);

  // Mask = (0.2, 0.3, 0.5)
  h_mask = {0.2f, 0.3f, 0.5f};

  CudaVector mask(h_mask);
  {
    CudaVector output(h_output);
    CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
    CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
    ASSERT_EQ(
        cudaBatchedLaplacianBlendWithContext3<float>(
            img1.data(),
            img2.data(),
            img3.data(),
            mask.data(),
            output.data(),
            ctx,
            channels,
            0),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  // Expected blending:
  // R = 0.2*10 + 0.3*50 + 0.5*90 = 2 + 15 + 45 = 62
  // G = 0.2*20 + 0.3*60 + 0.5*100 = 4 + 18 + 50 = 72
  // B = 0.2*30 + 0.3*70 + 0.5*110 = 6 + 21 + 55 = 82
  // A = 0.2*40 + 0.3*80 + 0.5*120 = 8 + 24 + 60 = 92
  std::vector<float> expected = {62.0f, 72.0f, 82.0f, 92.0f};

  for (int c = 0; c < channels; c++) {
    EXPECT_NEAR(h_output[c], expected[c], kEpsilon) << "Channel " << c << " mismatch for RGBA blending.";
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}

// Test 4: RGBA, alpha==0 in one image should skip that image (weights renormalized)
TEST(CudaBlend3SmallTest, AlphaZeroSkipsContribution) {
  const int width = 1;
  const int height = 1;
  const int channels = 4; // RGBA
  const int batchSize = 1;
  const int numLevels = 1;
  const int pixelCount = width * height * channels * batchSize;
  const int maskCount = width * height * 3;

  // Three RGBA pixels with distinct colors; image2 alpha=0 (fully transparent)
  std::vector<float> h_image1{10.0f, 20.0f, 30.0f, 255.0f};
  std::vector<float> h_image2{100.0f, 110.0f, 120.0f, 0.0f};
  std::vector<float> h_image3{90.0f, 100.0f, 110.0f, 255.0f};
  std::vector<float> h_mask{0.1f, 0.8f, 0.1f}; // mostly chooses image2, but it's alpha==0
  std::vector<float> h_output(pixelCount, 0.0f);

  CudaVector img1(h_image1), img2(h_image2), img3(h_image3);
  CudaVector mask(h_mask);
  CudaVector output(h_output);
  CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
  CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
  ASSERT_EQ(
      cudaBatchedLaplacianBlendWithContext3<float>(
          img1.data(),
          img2.data(),
          img3.data(),
          mask.data(),
          output.data(),
          ctx,
          channels,
          0),
      cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());
  // Pull result back to host for verification
  CUDA_CHECK(cudaMemcpy(h_output.data(), output.data(), pixelCount * sizeof(float), cudaMemcpyDeviceToHost));

  // Because image2 has alpha==0, its weight is zeroed and remaining weights renormalize:
  // m1' = 0.1/(0.1+0.1) = 0.5, m3' = 0.5. So RGB = (img1+img3)/2, A = (255+255)/2 = 255.
  std::vector<float> expected{(10.0f + 90.0f) * 0.5f,
                              (20.0f + 100.0f) * 0.5f,
                              (30.0f + 110.0f) * 0.5f,
                              255.0f};
  for (int c = 0; c < channels; ++c) {
    EXPECT_NEAR(h_output[c], expected[c], kEpsilon) << "Channel " << c << " mismatch with alpha gating.";
  }
}

// Test 5: If mask selects a transparent source (weight=1 on alpha==0),
// kernel should fallback to a non-transparent contributor (highest alpha)
TEST(CudaBlend3SmallTest, MaskSelectsTransparentThenFallback) {
  const int width = 1;
  const int height = 1;
  const int channels = 4; // RGBA
  const int batchSize = 1;
  const int numLevels = 1;
  const int pixelCount = width * height * channels * batchSize;

  // image1: solid red, alpha 255
  std::vector<float> h_image1{255.0f, 0.0f, 0.0f, 255.0f};
  // image2: green, but fully transparent
  std::vector<float> h_image2{0.0f, 255.0f, 0.0f, 0.0f};
  // image3: blue, lower alpha
  std::vector<float> h_image3{0.0f, 0.0f, 255.0f, 128.0f};
  // Mask selects only image2
  std::vector<float> h_mask{0.0f, 1.0f, 0.0f};
  std::vector<float> h_output(pixelCount, 0.0f);

  CudaVector img1(h_image1), img2(h_image2), img3(h_image3);
  CudaVector mask(h_mask);
  CudaVector output(h_output);
  CUDA_CHECK(cudaMemset(output.data(), 0, pixelCount * sizeof(float)));
  CudaBatchLaplacianBlendContext3<float> ctx(width, height, numLevels, batchSize);
  ASSERT_EQ(
      cudaBatchedLaplacianBlendWithContext3<float>(
          img1.data(), img2.data(), img3.data(), mask.data(), output.data(), ctx, channels, 0),
      cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output.data(), output.data(), pixelCount * sizeof(float), cudaMemcpyDeviceToHost));

  // Expect fallback picked image1 (highest alpha 255)
  std::vector<float> expected{255.0f, 0.0f, 0.0f, 255.0f};
  for (int c = 0; c < channels; ++c) {
    EXPECT_NEAR(h_output[c], expected[c], kEpsilon) << "Channel " << c << " mismatch in fallback.";
  }
}

// Test 6: Multi-level overlap seam (no alpha holes)
// Construct a 4x4 RGBA with full alpha; mask has top-half channel0, bottom-half channel1.
// With 2+ levels, blended alpha should remain 255 across seam and RGB should be non-zero.
TEST(CudaBlend3SmallTest, MultiLevelOverlapHasNoAlphaHoles) {
  const int W = 4, H = 4, C = 4, B = 1, L = 2; // two levels
  const int N = W * H * C;

  // image1 = red(100), image2 = green(150), image3 = blue(200); all alpha=255
  std::vector<float> i1(N, 0.0f), i2(N, 0.0f), i3(N, 0.0f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int idx = (y * W + x) * C;
      i1[idx + 0] = 100.0f; i1[idx + 3] = 255.0f; // R, A
      i2[idx + 1] = 150.0f; i2[idx + 3] = 255.0f; // G, A
      i3[idx + 2] = 200.0f; i3[idx + 3] = 255.0f; // B, A
    }
  }
  // Mask: channel0=1 for top half (y<2), channel1=1 for bottom (y>=2), channel2=0
  std::vector<float> m(W * H * 3, 0.0f);
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int mi = (y * W + x) * 3;
      if (y < H / 2) m[mi + 0] = 1.0f; else m[mi + 1] = 1.0f;
    }
  }
  std::vector<float> out(N, 0.0f);

  CudaVector d_i1(i1), d_i2(i2), d_i3(i3);
  CudaVector d_m(m);
  CudaVector d_out(out);
  CUDA_CHECK(cudaMemset(d_out.data(), 0, N * sizeof(float)));

  CudaBatchLaplacianBlendContext3<float> ctx(W, H, L, B);
  ASSERT_EQ(
      cudaBatchedLaplacianBlendWithContext3<float>(
          d_i1.data(), d_i2.data(), d_i3.data(), d_m.data(), d_out.data(), ctx, C, 0),
      cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(out.data(), d_out.data(), N * sizeof(float), cudaMemcpyDeviceToHost));

  // Check seam row (y==1 and y==2) — alpha must be 255; RGB should be from red/green mix, not zeros
  auto check_pixel = [&](int y, int x) {
    int idx = (y * W + x) * C;
    EXPECT_GE(out[idx + 3], 250.0f) << "Alpha hole at (" << x << "," << y << ")";
    float rgb_sum = out[idx + 0] + out[idx + 1] + out[idx + 2];
    EXPECT_GT(rgb_sum, 50.0f) << "RGB vanished at (" << x << "," << y << ")";
  };
  for (int x = 0; x < W; ++x) {
    check_pixel(1, x);
    check_pixel(2, x);
  }
}
