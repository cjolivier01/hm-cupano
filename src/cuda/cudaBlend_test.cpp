#include <cupano/gpu/gpu_runtime.h>
#include <gtest/gtest.h>
#include <cstring>

#include "cudaBlend.h"

#include <algorithm>
#include <array>
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

#if GPU_BACKEND_CUDA || GPU_BACKEND_HIP
TEST(CudaBlendLifetime, ContextCleanupAfterCallerStreamDestruction) {
  constexpr int W = 65, H = 33;
  const std::vector<float> input(W * H * 3, 7.0f);
  const std::vector<float> mask(W * H * 3, 1.0f / 3.0f);
  float *d_input, *d_mask, *d_output;
  CUDA_CHECK(cudaMalloc(&d_input, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_mask, mask.size() * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_output, input.size() * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_mask, mask.data(), mask.size() * sizeof(float), cudaMemcpyHostToDevice));

  for (int variant = 0; variant < 4; ++variant) {
    std::vector<float> reference;
    for (bool wait_before_destroy : {true, false}) {
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      {
        CudaBatchLaplacianBlendContext<float> two(W, H, 3, 1);
        CudaBatchLaplacianBlendContext3<float> three(W, H, 3, 1);
        CudaBatchLaplacianBlendContextN<float, 3> many(W, H, 3, 1);
        if (variant == 0) {
          CUDA_CHECK(
              (cudaBatchedLaplacianBlendWithContext<float, float>(d_input, d_input, d_mask, d_output, two, 3, stream)));
        } else if (variant == 1) {
          CUDA_CHECK((cudaBatchedLaplacianBlendWithContext3<float, float>(
              d_input, d_input, d_input, d_mask, d_output, three, 3, stream)));
        } else if (variant == 2) {
          CUDA_CHECK((cudaBatchedLaplacianBlendWithContextN<float, float, 3, 3>(
              {d_input, d_input, d_input}, d_mask, d_output, many, stream)));
        } else {
          CUDA_CHECK((cudaBatchedLaplacianBlendOptimized3<float, float>(
              d_input, d_input, d_input, d_mask, d_output, three, 3, stream)));
        }
        // Destroying a stream does not wait for its kernels. The contexts must
        // retain a dependency on that work without accessing the destroyed handle.
        if (wait_before_destroy)
          CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaStreamDestroy(stream));
      }
      CUDA_CHECK(cudaDeviceSynchronize());
      std::vector<float> output(input.size());
      CUDA_CHECK(cudaMemcpy(output.data(), d_output, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
      if (wait_before_destroy) {
        reference = output;
      } else {
        EXPECT_EQ(output, reference) << "variant=" << variant;
      }
      if (variant < 3) {
        for (float value : output)
          ASSERT_NEAR(value, 7.0f, 1e-4f) << "variant=" << variant;
      }
    }
  }
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_mask));
  CUDA_CHECK(cudaFree(d_output));
}
#endif

#if GPU_BACKEND_CUDA || GPU_BACKEND_HIP
TEST(CudaBlendOptimized3, MultiBlockPyramidsMatchCpuReference) {
  for (int channels : {3, 4}) {
    for (const auto size : {std::pair<int, int>{64, 32}, {65, 33}, {257, 65}}) {
      const int width = size.first, height = size.second, batch = 2, levels = 4;
      const size_t count = static_cast<size_t>(width) * height * channels * batch;
      cudaStream_t stream;
      CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
      {
        CudaBatchLaplacianBlendContext3<float> context(width, height, levels, batch);
        float *images[3], *mask, *output;
        for (auto& image : images)
          CUDA_CHECK(cudaMallocAsync(&image, count * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&mask, static_cast<size_t>(width) * height * 3 * sizeof(float), stream));
        CUDA_CHECK(cudaMallocAsync(&output, count * sizeof(float), stream));
        std::vector<float> host_mask(width * height * 3, 1.0f / 3.0f);
        CUDA_CHECK(
            cudaMemcpyAsync(mask, host_mask.data(), host_mask.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
        std::array<std::vector<float>, 3> host;
        for (int frame = 0; frame < 3; ++frame) {
          for (int camera = 0; camera < 3; ++camera) {
            host[camera].resize(count);
            for (size_t i = 0; i < count; ++i)
              host[camera][i] = 20.0f + static_cast<float>((i * 13 + camera * 17 + frame * 19) % 101);
            CUDA_CHECK(cudaMemcpyAsync(
                images[camera], host[camera].data(), count * sizeof(float), cudaMemcpyHostToDevice, stream));
          }
          if (frame) {
            // Expose reads of a neighboring block's unwritten next-level data,
            // rather than letting an allocation retain plausible values.
            for (int level = 1; level < context.numLevels; ++level) {
              const size_t bytes = static_cast<size_t>(context.widths[level]) * context.heights[level] * channels *
                  batch * sizeof(float);
              for (float* ptr : {context.d_gauss1[level], context.d_gauss2[level], context.d_gauss3[level]})
                CUDA_CHECK(cudaMemsetAsync(ptr, 0xff, bytes, stream));
            }
          }
          CUDA_CHECK((cudaBatchedLaplacianBlendOptimized3<float, float>(
              images[0], images[1], images[2], mask, output, context, channels, stream)));
          CUDA_CHECK(cudaStreamSynchronize(stream));

          const std::array<const std::vector<float*>*, 3> gauss = {
              &context.d_gauss1, &context.d_gauss2, &context.d_gauss3};
          const std::array<const std::vector<float*>*, 3> lap = {&context.d_lap1, &context.d_lap2, &context.d_lap3};
          for (int camera = 0; camera < 3; ++camera) {
            std::vector<float> current = host[camera];
            for (int level = 0; level < context.numLevels - 1; ++level) {
              const int w = context.widths[level], h = context.heights[level];
              const int nw = context.widths[level + 1], nh = context.heights[level + 1];
              std::vector<float> next(static_cast<size_t>(nw) * nh * channels * batch);
              for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < nh; ++y) {
                  for (int x = 0; x < nw; ++x) {
                    for (int c = 0; c < channels; ++c) {
                      float sum = 0;
                      int samples = 0;
                      for (int dy = 0; dy < 2; ++dy) {
                        for (int dx = 0; dx < 2; ++dx) {
                          if (2 * x + dx < w && 2 * y + dy < h) {
                            sum += current[((b * h + 2 * y + dy) * w + 2 * x + dx) * channels + c];
                            ++samples;
                          }
                        }
                      }
                      next[((b * nh + y) * nw + x) * channels + c] = sum / samples;
                    }
                  }
                }
              }
              std::vector<float> actual_next(next.size()), actual_lap(current.size());
              CUDA_CHECK(cudaMemcpyAsync(
                  actual_next.data(),
                  (*gauss[camera])[level + 1],
                  next.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              CUDA_CHECK(cudaMemcpyAsync(
                  actual_lap.data(),
                  (*lap[camera])[level],
                  current.size() * sizeof(float),
                  cudaMemcpyDeviceToHost,
                  stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              ASSERT_EQ(actual_next, next);
              for (int b = 0; b < batch; ++b) {
                for (int y = 0; y < h; ++y) {
                  for (int x = 0; x < w; ++x) {
                    const int x0 = x / 2, x1 = std::min(x0 + 1, nw - 1);
                    const int y0 = y / 2, y1 = std::min(y0 + 1, nh - 1);
                    const float dx = (x % 2) * 0.5f, dy = (y % 2) * 0.5f;
                    for (int c = 0; c < channels; ++c) {
                      auto value = [&](int xx, int yy) { return next[((b * nh + yy) * nw + xx) * channels + c]; };
                      const float up = (1 - dx) * (1 - dy) * value(x0, y0) + dx * (1 - dy) * value(x1, y0) +
                          (1 - dx) * dy * value(x0, y1) + dx * dy * value(x1, y1);
                      const size_t i = ((b * h + y) * w + x) * channels + c;
                      ASSERT_NEAR(actual_lap[i], current[i] - up, 1e-4f)
                          << "size=" << width << 'x' << height << " channels=" << channels << " frame=" << frame
                          << " camera=" << camera << " level=" << level << " batch=" << b << " pixel=" << x << ',' << y;
                    }
                  }
                }
              }
              current = std::move(next);
            }
          }
        }
        for (auto image : images)
          CUDA_CHECK(cudaFreeAsync(image, stream));
        CUDA_CHECK(cudaFreeAsync(mask, stream));
        CUDA_CHECK(cudaFreeAsync(output, stream));
      }
      CUDA_CHECK(cudaStreamSynchronize(stream));
      CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
}
#endif

#if GPU_BACKEND_CUDA || GPU_BACKEND_HIP
TEST(CudaBlendOptimized3, BytePyramidsDoNotReadPoisonedNeighbors) {
  constexpr int width = 65, height = 33, batch = 2, levels = 4;
  for (int channels : {3, 4}) {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    {
      CudaBatchLaplacianBlendContext3<unsigned char> context(width, height, levels, batch);
      const size_t count = static_cast<size_t>(width) * height * channels * batch;
      unsigned char *images[3], *mask, *output;
      std::array<std::vector<unsigned char>, 3> host;
      for (int camera = 0; camera < 3; ++camera) {
        host[camera].resize(count);
        for (int b = 0; b < batch; ++b)
          std::fill_n(host[camera].begin() + b * count / batch, count / batch, 20 + camera * 15 + b * 25);
        CUDA_CHECK(cudaMallocAsync(&images[camera], count, stream));
        CUDA_CHECK(cudaMemcpyAsync(images[camera], host[camera].data(), count, cudaMemcpyHostToDevice, stream));
      }
      std::vector<unsigned char> host_mask(width * height * 3, 0);
      for (size_t i = 0; i < host_mask.size(); i += 3)
        host_mask[i] = 1;
      CUDA_CHECK(cudaMallocAsync(&mask, host_mask.size(), stream));
      CUDA_CHECK(cudaMemcpyAsync(mask, host_mask.data(), host_mask.size(), cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMallocAsync(&output, count, stream));
      const std::array<const std::vector<unsigned char*>*, 3> gauss = {
          &context.d_gauss1, &context.d_gauss2, &context.d_gauss3};
      const std::array<const std::vector<unsigned char*>*, 3> lap = {&context.d_lap1, &context.d_lap2, &context.d_lap3};
      for (int frame = 0; frame < 2; ++frame) {
        if (frame) {
          for (int camera = 0; camera < 3; ++camera) {
            for (int level = 1; level < levels; ++level) {
              const size_t bytes =
                  static_cast<size_t>(context.widths[level]) * context.heights[level] * channels * batch;
              CUDA_CHECK(cudaMemsetAsync((*gauss[camera])[level], 0xff, bytes, stream));
            }
          }
        }
        CUDA_CHECK((cudaBatchedLaplacianBlendOptimized3<unsigned char, float>(
            images[0], images[1], images[2], mask, output, context, channels, stream)));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int camera = 0; camera < 3; ++camera) {
          for (int level = 0; level < levels; ++level) {
            const size_t bytes = static_cast<size_t>(context.widths[level]) * context.heights[level] * channels * batch;
            std::vector<unsigned char> actual(bytes), expected(bytes);
            for (int b = 0; b < batch; ++b)
              std::fill_n(expected.begin() + b * bytes / batch, bytes / batch, 20 + camera * 15 + b * 25);
            CUDA_CHECK(cudaMemcpyAsync(actual.data(), (*gauss[camera])[level], bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            ASSERT_EQ(actual, expected) << "camera=" << camera << " level=" << level;
            if (level < levels - 1) {
              CUDA_CHECK(cudaMemcpyAsync(actual.data(), (*lap[camera])[level], bytes, cudaMemcpyDeviceToHost, stream));
              CUDA_CHECK(cudaStreamSynchronize(stream));
              ASSERT_EQ(actual, std::vector<unsigned char>(bytes, 0)) << "camera=" << camera << " level=" << level;
            }
          }
        }
      }
      for (auto image : images)
        CUDA_CHECK(cudaFreeAsync(image, stream));
      CUDA_CHECK(cudaFreeAsync(mask, stream));
      CUDA_CHECK(cudaFreeAsync(output, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }
}
#endif
