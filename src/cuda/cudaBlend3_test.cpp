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

#include <cuda_runtime.h>
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
    assert(cuerr == cudaError_t::cudaSuccess);
    cuerr = cudaMemcpy(d_ptr, vec_.data(), vec_.size() * sizeof(Type), cudaMemcpyKind::cudaMemcpyHostToDevice);
    assert(cuerr == cudaError_t::cudaSuccess);
  }
  ~CudaVector() {
    if (d_ptr) {
      cudaError_t cuerr =
          cudaMemcpy(vec_.data(), d_ptr, vec_.size() * sizeof(Type), cudaMemcpyKind::cudaMemcpyDeviceToHost);
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
    // Synchronize to ensure the host copy is done
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
