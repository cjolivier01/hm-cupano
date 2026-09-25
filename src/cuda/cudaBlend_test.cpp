#include <cupano/gpu/gpu_runtime.h>
#include <gtest/gtest.h>
#include <cstring>

#include "cudaBlend.h"

#include <cuda_fp16.h>

#include <algorithm>
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

TEST(CudaBlendSmallTest, DisabledMaskPyramidCacheTracksInPlaceUpdates) {
  constexpr int width = 4;
  constexpr int height = 4;
  constexpr int channels = 3;
  constexpr int batch_size = 1;
  constexpr int num_levels = 2;
  constexpr int image_value_count = width * height * channels * batch_size;
  constexpr int mask_value_count = width * height;

  const std::vector<float> h_image1(image_value_count, 10.0f);
  const std::vector<float> h_image2(image_value_count, 100.0f);
  std::vector<float> h_mask(mask_value_count, 1.0f);
  std::vector<float> h_output(image_value_count, 0.0f);

  float* d_image1 = nullptr;
  float* d_image2 = nullptr;
  float* d_mask = nullptr;
  float* d_output = nullptr;
  CUDA_CHECK(cudaMalloc(&d_image1, h_image1.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_image2, h_image2.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, h_mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_image1, h_image1.data(), h_image1.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_image2, h_image2.data(), h_image2.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(float), cudaMemcpyHostToDevice));

  {
    CudaBatchLaplacianBlendContext<float> ctx(width, height, num_levels, batch_size);
    ASSERT_EQ(
        (cudaBatchedLaplacianBlendWithContext<float, float>(
            d_image1, d_image2, d_mask, d_output, ctx, channels, cudaStream_t{})),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : h_output) {
      EXPECT_NEAR(value, 10.0f, 1e-5f);
    }

    std::fill(h_mask.begin(), h_mask.end(), 0.0f);
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask.data(), h_mask.size() * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(
        (cudaBatchedLaplacianBlendWithContext<float, float>(
            d_image1,
            d_image2,
            d_mask,
            d_output,
            ctx,
            channels,
            cudaStream_t{},
            /*cacheMaskPyramid=*/false)),
        cudaSuccess);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, h_output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    for (float value : h_output) {
      EXPECT_NEAR(value, 100.0f, 1e-5f);
    }
  }

  cudaFree(d_image1);
  cudaFree(d_image2);
  cudaFree(d_mask);
  cudaFree(d_output);
}

namespace {
template <typename T>
void check_compact_workspace() {
  for (int channels : {3, 4}) {
    for (int levels : {1, 6}) {
      constexpr int w = 65, h = 49, batch = 2;
      const size_t values = w * h * batch * channels;
      const size_t bytes = values * sizeof(T);
      std::vector<T> left(values), right(values), mask(w * h), reference(values), compact(values);
      T *d_left = nullptr, *d_right = nullptr, *d_mask = nullptr, *d_reference = nullptr, *d_compact = nullptr,
        *d_alternate = nullptr;
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      CUDA_CHECK(cudaMalloc(&d_left, bytes));
      CUDA_CHECK(cudaMalloc(&d_right, bytes));
      CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&d_reference, bytes));
      CUDA_CHECK(cudaMalloc(&d_compact, bytes));
      CUDA_CHECK(cudaMalloc(&d_alternate, bytes));
      {
        CudaBatchLaplacianBlendContext<T> normal(w, h, levels, batch);
        CudaBatchLaplacianBlendContext<T> lean(w, h, levels, batch, true);
        for (int frame = 0; frame < 4; ++frame) {
          for (size_t i = 0; i < values; ++i) {
            left[i] = static_cast<T>(float((i * 17 + frame * 41) % 1024) / 4.0f);
            right[i] = static_cast<T>(float((i * 31 + frame * 57) % 1024) / 4.0f);
            if (channels == 4 && i % 4 == 3) {
              left[i] = static_cast<T>(float((i + frame) % 5 ? 255 : 0));
              right[i] = static_cast<T>(float((i + frame) % 7 ? 255 : 0));
            }
          }
          for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = static_cast<T>(float((i + frame * 11) % 101) / 100.0f);
          CUDA_CHECK(cudaMemcpyAsync(d_left, left.data(), bytes, cudaMemcpyHostToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(d_right, right.data(), bytes, cudaMemcpyHostToDevice, stream));
          CUDA_CHECK(cudaMemcpyAsync(d_mask, mask.data(), mask.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
          ASSERT_EQ(
              (cudaBatchedLaplacianBlendWithContext<T, float>(
                  d_left, d_right, d_mask, d_reference, normal, channels, stream, false)),
              cudaSuccess);
          ASSERT_EQ(
              (cudaBatchedLaplacianBlendWithContext<T, float>(
                  d_left,
                  d_right,
                  d_mask,
                  frame % 3 == 0 ? d_left : (frame % 3 == 1 ? d_compact : d_alternate),
                  lean,
                  channels,
                  stream,
                  false)),
              cudaSuccess);
          CUDA_CHECK(cudaMemcpyAsync(reference.data(), d_reference, bytes, cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(
              compact.data(),
              frame % 3 == 0 ? d_left : (frame % 3 == 1 ? d_compact : d_alternate),
              bytes,
              cudaMemcpyDeviceToHost,
              stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          ASSERT_EQ(std::memcmp(reference.data(), compact.data(), bytes), 0)
              << "channels=" << channels << " levels=" << levels << " frame=" << frame;
          ASSERT_LT(lean.allocation_size, normal.allocation_size / 4);
        }
      }
      CUDA_CHECK(cudaFree(d_left));
      CUDA_CHECK(cudaFree(d_right));
      CUDA_CHECK(cudaFree(d_mask));
      CUDA_CHECK(cudaFree(d_reference));
      CUDA_CHECK(cudaFree(d_compact));
      CUDA_CHECK(cudaFree(d_alternate));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
}
} // namespace
TEST(CudaBlendSmallTest, CompactWorkspaceIsBitExactFloat) {
  check_compact_workspace<float>();
}
TEST(CudaBlendSmallTest, CompactWorkspaceIsBitExactHalf) {
  check_compact_workspace<__half>();
}

#include <type_traits>
#include "cupano/cuda/cudaBlend3.h"
#include "cupano/cuda/cudaBlendN.h"

namespace {
template <typename T, int N, bool three>
void check_multi_compact_workspace() {
  for (int channels : {3, 4}) {
    for (int levels : {1, 5}) {
      constexpr int w = 65, h = 49, batch = 2;
      const size_t values = w * h * batch * channels, bytes = values * sizeof(T);
      std::array<std::vector<T>, N> host;
      std::array<T*, N> device{};
      std::vector<const T*> pointers(N);
      std::vector<T> mask(w * h * N), reference(values), compact(values);
      T *d_mask = nullptr, *d_reference = nullptr, *d_compact = nullptr, *d_alternate = nullptr;
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreate(&stream));
      for (int i = 0; i < N; ++i) {
        host[i].resize(values);
        CUDA_CHECK(cudaMalloc(&device[i], bytes));
        pointers[i] = device[i];
      }
      CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(T)));
      CUDA_CHECK(cudaMalloc(&d_reference, bytes));
      CUDA_CHECK(cudaMalloc(&d_compact, bytes));
      CUDA_CHECK(cudaMalloc(&d_alternate, bytes));
      {
        using Context =
            std::conditional_t<three, CudaBatchLaplacianBlendContext3<T>, CudaBatchLaplacianBlendContextN<T, N>>;
        Context normal(w, h, levels, batch), lean(w, h, levels, batch, true);
        auto blend = [&](Context& ctx, T* output) {
          if constexpr (three) {
            return cudaBatchedLaplacianBlendWithContext3<T, float>(
                pointers[0], pointers[1], pointers[2], d_mask, output, ctx, channels, stream, false);
          } else {
            if (channels == 3)
              return cudaBatchedLaplacianBlendWithContextN<T, float, N, 3>(
                  pointers, d_mask, output, ctx, stream, false);
            return cudaBatchedLaplacianBlendWithContextN<T, float, N, 4>(pointers, d_mask, output, ctx, stream, false);
          }
        };
        for (int frame = 0; frame < 4; ++frame) {
          for (int camera = 0; camera < N; ++camera) {
            for (size_t i = 0; i < values; ++i) {
              host[camera][i] = static_cast<T>(float((i * 17 + camera * 131 + frame * 41) % 1024) / 4.0f);
              if (channels == 4 && i % 4 == 3)
                host[camera][i] = static_cast<T>(float((i + camera + frame) % 5 ? 255 : 0));
            }
            CUDA_CHECK(cudaMemcpyAsync(device[camera], host[camera].data(), bytes, cudaMemcpyHostToDevice, stream));
          }
          for (size_t i = 0; i < mask.size(); ++i)
            mask[i] = static_cast<T>(float((i + frame * 11) % 101) / 100.0f);
          CUDA_CHECK(cudaMemcpyAsync(d_mask, mask.data(), mask.size() * sizeof(T), cudaMemcpyHostToDevice, stream));
          if constexpr (!three) {
            if (frame)
              std::rotate(pointers.begin(), pointers.begin() + 1, pointers.end());
          }
          T* output = frame % 2 ? const_cast<T*>(pointers[0]) : d_compact;
          ASSERT_EQ(blend(normal, d_reference), cudaSuccess);
          ASSERT_EQ(blend(lean, output), cudaSuccess);
          CUDA_CHECK(cudaMemcpyAsync(reference.data(), d_reference, bytes, cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaMemcpyAsync(compact.data(), output, bytes, cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
          ASSERT_EQ(std::memcmp(reference.data(), compact.data(), bytes), 0)
              << "channels=" << channels << " levels=" << levels << " frame=" << frame;
          ASSERT_LT(lean.allocation_size, normal.allocation_size / 3);
        }
      }
      for (T* p : device)
        CUDA_CHECK(cudaFree(p));
      CUDA_CHECK(cudaFree(d_mask));
      CUDA_CHECK(cudaFree(d_reference));
      CUDA_CHECK(cudaFree(d_compact));
      CUDA_CHECK(cudaFree(d_alternate));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
}
} // namespace
TEST(CudaBlendSmallTest, ThreeCompactWorkspaceIsBitExactFloat) {
  check_multi_compact_workspace<float, 3, true>();
}
TEST(CudaBlendSmallTest, ThreeCompactWorkspaceIsBitExactHalf) {
  check_multi_compact_workspace<__half, 3, true>();
}
TEST(CudaBlendSmallTest, NCompactWorkspaceIsBitExactFloat) {
  check_multi_compact_workspace<float, 5, false>();
}
TEST(CudaBlendSmallTest, NCompactWorkspaceIsBitExactHalf) {
  check_multi_compact_workspace<__half, 5, false>();
}
