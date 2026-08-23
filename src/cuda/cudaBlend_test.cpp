#include <cupano/gpu/gpu_runtime.h>
#include <gtest/gtest.h>

#include "cudaBlend.h"

#include <cuda_fp16.h>

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

namespace {

std::vector<__half> to_half(const std::vector<float>& values) {
  std::vector<__half> out(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    out[i] = __float2half(values[i]);
  }
  return out;
}

} // namespace

TEST(CudaBlendSmallTest, HalfRgbaAlphaZeroSkipsContribution) {
  constexpr int width = 1;
  constexpr int height = 1;
  constexpr int channels = 4;
  constexpr int batch_size = 1;
  constexpr int num_levels = 1;
  constexpr int pixel_count = width * height * channels * batch_size;

  std::vector<__half> h_image1 = to_half({10.0f, 20.0f, 30.0f, 0.0f});
  std::vector<__half> h_image2 = to_half({100.0f, 110.0f, 120.0f, 255.0f});
  std::vector<__half> h_mask = to_half({0.9f});
  std::vector<__half> h_output(pixel_count, __float2half(0.0f));

  __half* d_image1 = nullptr;
  __half* d_image2 = nullptr;
  __half* d_mask = nullptr;
  __half* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_image1, h_image1.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_image2, h_image2.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(__half)));
  CUDA_CHECK(cudaMemcpy(d_image1, h_image1.data(), h_image1.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_image2, h_image2.data(), h_image2.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(d_output, 0, h_output.size() * sizeof(__half)));

  CudaBatchLaplacianBlendContext<__half> ctx(width, height, num_levels, batch_size);
  ASSERT_EQ(
      (cudaBatchedLaplacianBlendWithContext<__half, float>(
          d_image1, d_image2, d_mask, d_output, ctx, channels, cudaStream_t{})),
      cudaSuccess);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(__half), cudaMemcpyDeviceToHost));

  const std::vector<float> expected{100.0f, 110.0f, 120.0f, 255.0f};
  for (int c = 0; c < channels; ++c) {
    EXPECT_NEAR(__half2float(h_output[c]), expected[c], 0.01f) << "Channel " << c << " mismatch.";
  }

  cudaFree(d_image1);
  cudaFree(d_image2);
  cudaFree(d_mask);
  cudaFree(d_output);
}
