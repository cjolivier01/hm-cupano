#include "cudaBlend.h"
#include "cudaBlend3.h"
#include "cudaBlendN.h"
#include "cudaFusedKernels3.h"
#include "cudaImageAdjust.cuh"
#include "cudaMakeFull.h"
#include "cudaRemap.h"

#include <cupano/pano/canvasManager3.h>

#if defined(__has_include)
#if __has_include(<vulkan/vulkan.h>)
#include <vulkan/vulkan.h>
#endif
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
struct PixelTraits;

template <>
struct PixelTraits<unsigned char> {
  static constexpr int kChannels = 1;
  static float get(const unsigned char& p, int) { return static_cast<float>(p); }
  static void set(unsigned char& p, int, float v) {
    p = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
  }
};

template <>
struct PixelTraits<uchar1> {
  static constexpr int kChannels = 1;
  static float get(const uchar1& p, int) { return static_cast<float>(p.x); }
  static void set(uchar1& p, int, float v) {
    p.x = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
  }
};

template <>
struct PixelTraits<uchar3> {
  static constexpr int kChannels = 3;
  static float get(const uchar3& p, int c) {
    switch (c) {
      case 0: return static_cast<float>(p.x);
      case 1: return static_cast<float>(p.y);
      default: return static_cast<float>(p.z);
    }
  }
  static void set(uchar3& p, int c, float v) {
    const auto cv = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
    switch (c) {
      case 0: p.x = cv; break;
      case 1: p.y = cv; break;
      default: p.z = cv; break;
    }
  }
};

template <>
struct PixelTraits<uchar4> {
  static constexpr int kChannels = 4;
  static float get(const uchar4& p, int c) {
    switch (c) {
      case 0: return static_cast<float>(p.x);
      case 1: return static_cast<float>(p.y);
      case 2: return static_cast<float>(p.z);
      default: return static_cast<float>(p.w);
    }
  }
  static void set(uchar4& p, int c, float v) {
    const auto cv = static_cast<unsigned char>(std::clamp(v, 0.0f, 255.0f));
    switch (c) {
      case 0: p.x = cv; break;
      case 1: p.y = cv; break;
      case 2: p.z = cv; break;
      default: p.w = cv; break;
    }
  }
};

template <>
struct PixelTraits<float> {
  static constexpr int kChannels = 1;
  static float get(const float& p, int) { return p; }
  static void set(float& p, int, float v) { p = v; }
};

template <>
struct PixelTraits<float1> {
  static constexpr int kChannels = 1;
  static float get(const float1& p, int) { return p.x; }
  static void set(float1& p, int, float v) { p.x = v; }
};

template <>
struct PixelTraits<float3> {
  static constexpr int kChannels = 3;
  static float get(const float3& p, int c) {
    switch (c) {
      case 0: return p.x;
      case 1: return p.y;
      default: return p.z;
    }
  }
  static void set(float3& p, int c, float v) {
    switch (c) {
      case 0: p.x = v; break;
      case 1: p.y = v; break;
      default: p.z = v; break;
    }
  }
};

template <>
struct PixelTraits<float4> {
  static constexpr int kChannels = 4;
  static float get(const float4& p, int c) {
    switch (c) {
      case 0: return p.x;
      case 1: return p.y;
      case 2: return p.z;
      default: return p.w;
    }
  }
  static void set(float4& p, int c, float v) {
    switch (c) {
      case 0: p.x = v; break;
      case 1: p.y = v; break;
      case 2: p.z = v; break;
      default: p.w = v; break;
    }
  }
};

template <typename T>
T zero_pixel() {
  return T{};
}

template <typename T_out, typename T_in>
T_out pixel_cast(const T_in& in) {
  T_out out{};
  constexpr int in_ch = PixelTraits<T_in>::kChannels;
  constexpr int out_ch = PixelTraits<T_out>::kChannels;
  const int common = std::min(in_ch, out_ch);
  for (int c = 0; c < common; ++c) {
    PixelTraits<T_out>::set(out, c, PixelTraits<T_in>::get(in, c));
  }
  for (int c = common; c < out_ch; ++c) {
    float fill = 0.0f;
    if (c == 3) {
      fill = 255.0f;
    }
    PixelTraits<T_out>::set(out, c, fill);
  }
  return out;
}

template <typename T>
T apply_adjustment(const T& in, const float3& adj) {
  T out = in;
  if constexpr (PixelTraits<T>::kChannels >= 1) {
    PixelTraits<T>::set(out, 0, PixelTraits<T>::get(in, 0) + adj.x);
  }
  if constexpr (PixelTraits<T>::kChannels >= 2) {
    PixelTraits<T>::set(out, 1, PixelTraits<T>::get(in, 1) + adj.y);
  }
  if constexpr (PixelTraits<T>::kChannels >= 3) {
    PixelTraits<T>::set(out, 2, PixelTraits<T>::get(in, 2) + adj.z);
  }
  return out;
}

template <typename T>
T* surface_ptr_mut(const CudaSurface<T>& surf, int batch, int x, int y) {
  auto* base = reinterpret_cast<std::uint8_t*>(surf.d_ptr);
  const std::size_t batch_off = static_cast<std::size_t>(batch) * surf.pitch * surf.height;
  const std::size_t row_off = static_cast<std::size_t>(y) * surf.pitch;
  return reinterpret_cast<T*>(base + batch_off + row_off) + x;
}

template <typename T>
const T* surface_ptr_const(const CudaSurface<T>& surf, int batch, int x, int y) {
  const auto* base = reinterpret_cast<const std::uint8_t*>(surf.d_ptr);
  const std::size_t batch_off = static_cast<std::size_t>(batch) * surf.pitch * surf.height;
  const std::size_t row_off = static_cast<std::size_t>(y) * surf.pitch;
  return reinterpret_cast<const T*>(base + batch_off + row_off) + x;
}

inline void init_level_dims(std::vector<int>& widths, std::vector<int>& heights, int w, int h) {
  for (std::size_t i = 0; i < widths.size(); ++i) {
    widths[i] = std::max(1, w >> static_cast<int>(i));
    heights[i] = std::max(1, h >> static_cast<int>(i));
  }
}

template <typename T>
void blend_two_images(
    const T* img1,
    const T* img2,
    const T* mask,
    T* out,
    int width,
    int height,
    int channels,
    int batch) {
  const int pixels = width * height;
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < pixels; ++i) {
      const int pix_off = ((b * pixels) + i) * channels;
      const float w0_raw = static_cast<float>(mask[i]);
      float w0 = w0_raw;
      float w1 = 1.0f - w0_raw;

      if (channels == 4) {
        const float a0 = static_cast<float>(img1[pix_off + 3]);
        const float a1 = static_cast<float>(img2[pix_off + 3]);
        if (a0 <= 0.0f) w0 = 0.0f;
        if (a1 <= 0.0f) w1 = 0.0f;
        const float s = w0 + w1;
        if (s > 1e-8f) {
          w0 /= s;
          w1 /= s;
          for (int c = 0; c < 3; ++c) {
            out[pix_off + c] = static_cast<T>(w0 * img1[pix_off + c] + w1 * img2[pix_off + c]);
          }
          out[pix_off + 3] = static_cast<T>(std::max(a0, a1));
        } else {
          const bool choose0 = a0 >= a1;
          for (int c = 0; c < 4; ++c) {
            out[pix_off + c] = choose0 ? img1[pix_off + c] : img2[pix_off + c];
          }
        }
      } else {
        for (int c = 0; c < channels; ++c) {
          out[pix_off + c] = static_cast<T>(w0 * img1[pix_off + c] + w1 * img2[pix_off + c]);
        }
      }
    }
  }
}

template <typename T>
void blend_three_images(
    const T* img0,
    const T* img1,
    const T* img2,
    const T* mask3,
    T* out,
    int width,
    int height,
    int channels,
    int batch) {
  const int pixels = width * height;
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < pixels; ++i) {
      const int pix_off = ((b * pixels) + i) * channels;
      const int moff = i * 3;
      float w[3] = {static_cast<float>(mask3[moff + 0]), static_cast<float>(mask3[moff + 1]), static_cast<float>(mask3[moff + 2])};

      const T* imgs[3] = {img0, img1, img2};

      if (channels == 4) {
        float a[3] = {
            static_cast<float>(img0[pix_off + 3]),
            static_cast<float>(img1[pix_off + 3]),
            static_cast<float>(img2[pix_off + 3]),
        };
        for (int k = 0; k < 3; ++k) {
          if (a[k] <= 0.0f) {
            w[k] = 0.0f;
          }
        }
        const float s = w[0] + w[1] + w[2];
        if (s > 1e-8f) {
          w[0] /= s;
          w[1] /= s;
          w[2] /= s;
          for (int c = 0; c < 3; ++c) {
            const float v =
                w[0] * img0[pix_off + c] + w[1] * img1[pix_off + c] + w[2] * img2[pix_off + c];
            out[pix_off + c] = static_cast<T>(v);
          }
          out[pix_off + 3] = static_cast<T>(std::max({a[0], a[1], a[2]}));
        } else {
          int best = 0;
          if (a[1] > a[best]) best = 1;
          if (a[2] > a[best]) best = 2;
          for (int c = 0; c < 4; ++c) {
            out[pix_off + c] = imgs[best][pix_off + c];
          }
        }
      } else {
        for (int c = 0; c < channels; ++c) {
          const float v = w[0] * img0[pix_off + c] + w[1] * img1[pix_off + c] + w[2] * img2[pix_off + c];
          out[pix_off + c] = static_cast<T>(v);
        }
      }
    }
  }
}

template <typename T, int N_IMAGES, int CHANNELS>
void blend_n_images(
    const std::vector<const T*>& imgs,
    const T* mask,
    T* out,
    int width,
    int height,
    int batch) {
  const int pixels = width * height;
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < pixels; ++i) {
      const int pix_off = ((b * pixels) + i) * CHANNELS;
      float w[N_IMAGES];
      for (int k = 0; k < N_IMAGES; ++k) {
        w[k] = static_cast<float>(mask[i * N_IMAGES + k]);
      }

      if constexpr (CHANNELS == 4) {
        float a[N_IMAGES];
        for (int k = 0; k < N_IMAGES; ++k) {
          a[k] = static_cast<float>(imgs[k][pix_off + 3]);
          if (a[k] <= 0.0f) {
            w[k] = 0.0f;
          }
        }
        float s = 0.0f;
        for (int k = 0; k < N_IMAGES; ++k) s += w[k];
        if (s > 1e-8f) {
          for (int k = 0; k < N_IMAGES; ++k) w[k] /= s;
          for (int c = 0; c < 3; ++c) {
            float v = 0.0f;
            for (int k = 0; k < N_IMAGES; ++k) {
              v += w[k] * imgs[k][pix_off + c];
            }
            out[pix_off + c] = static_cast<T>(v);
          }
          float amax = a[0];
          for (int k = 1; k < N_IMAGES; ++k) amax = std::max(amax, a[k]);
          out[pix_off + 3] = static_cast<T>(amax);
        } else {
          int best = 0;
          float best_a = a[0];
          for (int k = 1; k < N_IMAGES; ++k) {
            if (a[k] > best_a) {
              best = k;
              best_a = a[k];
            }
          }
          for (int c = 0; c < 4; ++c) {
            out[pix_off + c] = imgs[best][pix_off + c];
          }
        }
      } else {
        for (int c = 0; c < CHANNELS; ++c) {
          float v = 0.0f;
          for (int k = 0; k < N_IMAGES; ++k) {
            v += w[k] * imgs[k][pix_off + c];
          }
          out[pix_off + c] = static_cast<T>(v);
        }
      }
    }
  }
}

template <typename T_in, typename T_out>
cudaError_t remap_core(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    bool no_unmapped_write,
    bool filter_by_dest_map,
    int this_image_index,
    const unsigned char* dest_image_map,
    bool apply_adjust,
    const float3& adjustment) {
  if (!src.d_ptr || !dest.d_ptr || !mapX || !mapY) {
    return cudaErrorInvalidDevicePointer;
  }

  const int x_end = std::min(remapW, roiX + roiW);
  const int y_end = std::min(remapH, roiY + roiH);

  for (int y = std::max(0, roiY); y < y_end; ++y) {
    for (int x = std::max(0, roiX); x < x_end; ++x) {
      const int dx = offsetX + x;
      const int dy = offsetY + y;
      if (dx < 0 || dy < 0 || dx >= static_cast<int>(dest.width) || dy >= static_cast<int>(dest.height)) {
        continue;
      }
      if (filter_by_dest_map) {
        if (!dest_image_map) {
          return cudaErrorInvalidValue;
        }
        if (dest_image_map[dy * static_cast<int>(dest.width) + dx] != this_image_index) {
          continue;
        }
      }

      const unsigned short sx = mapX[y * remapW + x];
      const unsigned short sy = mapY[y * remapW + x];
      const bool mapped = sx < src.width && sy < src.height;

      for (int b = 0; b < batchSize; ++b) {
        T_out out_px = pixel_cast<T_out>(deflt);
        bool do_write = true;
        if (mapped) {
          T_in src_px = *surface_ptr_const(src, b, static_cast<int>(sx), static_cast<int>(sy));
          if (apply_adjust) {
            src_px = apply_adjustment(src_px, adjustment);
          }
          out_px = pixel_cast<T_out>(src_px);
        } else if (no_unmapped_write) {
          do_write = false;
        }

        if (do_write) {
          *surface_ptr_mut(dest, b, dx, dy) = out_px;
        }
      }
    }
  }

  return cudaSuccess;
}

} // namespace

// -----------------------------------------------------------------------------
// Image adjustment
// -----------------------------------------------------------------------------

template <typename T>
void adjustImageCudaBatch(T* d_image, int batchSize, int width, int height, const float3& adjustment) {
  if (!d_image) {
    return;
  }
  constexpr int kChannels = PixelTraits<T>::kChannels;
  const int pixels = batchSize * width * height;
  for (int i = 0; i < pixels; ++i) {
    d_image[i] = apply_adjustment(d_image[i], adjustment);
  }
  (void)kChannels;
}

// -----------------------------------------------------------------------------
// Make full / ROI copy
// -----------------------------------------------------------------------------

template <typename T>
cudaError_t AlphaConditionalCopy(CudaSurface<T>& image1, CudaSurface<T>& image2, int batchSize, cudaStream_t) {
  if (!image1.d_ptr || !image2.d_ptr) {
    return cudaErrorInvalidDevicePointer;
  }
  for (int b = 0; b < batchSize; ++b) {
    for (int y = 0; y < static_cast<int>(image1.height); ++y) {
      for (int x = 0; x < static_cast<int>(image1.width); ++x) {
        T* p1 = surface_ptr_mut(image1, b, x, y);
        T* p2 = surface_ptr_mut(image2, b, x, y);
        const float a1 = PixelTraits<T>::get(*p1, 3);
        const float a2 = PixelTraits<T>::get(*p2, 3);
        if (a1 == 0.0f && a2 != 0.0f) {
          *p1 = *p2;
        } else if (a2 == 0.0f && a1 != 0.0f) {
          *p2 = *p1;
        }
      }
    }
  }
  return cudaSuccess;
}

template <typename T_in, typename T_out>
cudaError_t copy_roi_batched(
    const CudaSurface<T_in>& src,
    int regionWidth,
    int regionHeight,
    int srcROI_x,
    int srcROI_y,
    CudaSurface<T_out> dest,
    int offsetX,
    int offsetY,
    int batchSize,
    cudaStream_t) {
  if (!src.d_ptr || !dest.d_ptr) {
    return cudaErrorInvalidDevicePointer;
  }

  for (int b = 0; b < batchSize; ++b) {
    for (int y = 0; y < regionHeight; ++y) {
      const int sy = srcROI_y + y;
      const int dy = offsetY + y;
      if (sy < 0 || dy < 0 || sy >= static_cast<int>(src.height) || dy >= static_cast<int>(dest.height)) {
        continue;
      }
      for (int x = 0; x < regionWidth; ++x) {
        const int sx = srcROI_x + x;
        const int dx = offsetX + x;
        if (sx < 0 || dx < 0 || sx >= static_cast<int>(src.width) || dx >= static_cast<int>(dest.width)) {
          continue;
        }
        *surface_ptr_mut(dest, b, dx, dy) = pixel_cast<T_out>(*surface_ptr_const(src, b, sx, sy));
      }
    }
  }

  return cudaSuccess;
}

template <typename T_in, typename T_out>
cudaError_t simple_make_full_batch(
    const CudaSurface<T_in>& src,
    int region_width,
    int region_height,
    int src_roi_x,
    int src_roi_y,
    int destOffsetX,
    int destOffsetY,
    bool,
    int batchSize,
    CudaSurface<T_out> dest,
    cudaStream_t stream) {
  if (!dest.d_ptr) {
    return cudaErrorInvalidDevicePointer;
  }
  cudaError_t cuerr = cudaMemsetAsync(dest.d_ptr, 0, total_size(dest, batchSize), stream);
  if (cuerr != cudaSuccess) {
    return cuerr;
  }
  return copy_roi_batched(
      src,
      region_width,
      region_height,
      src_roi_x,
      src_roi_y,
      dest,
      destOffsetX,
      destOffsetY,
      batchSize,
      stream);
}

// -----------------------------------------------------------------------------
// Remap
// -----------------------------------------------------------------------------

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in dflt,
    int batchSize,
    cudaStream_t stream) {
  CudaSurface<T_in> src{const_cast<T_in*>(d_src), static_cast<std::uint32_t>(srcW), static_cast<std::uint32_t>(srcH),
                        static_cast<std::uint32_t>(srcW * sizeof(T_in))};
  CudaSurface<T_out> dest{d_dest, static_cast<std::uint32_t>(destW), static_cast<std::uint32_t>(destH),
                          static_cast<std::uint32_t>(destW * sizeof(T_out))};
  return batched_remap_kernel_ex_offset(
      src, dest, d_mapX, d_mapY, dflt, batchSize, destW, destH, 0, 0, false, stream);
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write,
    cudaStream_t) {
  return remap_core(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      0,
      0,
      remapW,
      remapH,
      no_unmapped_write,
      false,
      0,
      nullptr,
      false,
      make_float3(0.0f, 0.0f, 0.0f));
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_roi(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    bool no_unmapped_write,
    cudaStream_t) {
  return remap_core(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      roiX,
      roiY,
      roiW,
      roiH,
      no_unmapped_write,
      false,
      0,
      nullptr,
      false,
      make_float3(0.0f, 0.0f, 0.0f));
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_adjust(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write,
    float3 adjustment,
    cudaStream_t) {
  return remap_core(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      0,
      0,
      remapW,
      remapH,
      no_unmapped_write,
      false,
      0,
      nullptr,
      true,
      adjustment);
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    cudaStream_t) {
  return remap_core(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      0,
      0,
      remapW,
      remapH,
      false,
      true,
      this_image_index,
      dest_image_map,
      false,
      make_float3(0.0f, 0.0f, 0.0f));
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust(
    const CudaSurface<T_in>& src,
    const CudaSurface<T_out>& dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    float3 adjustment,
    cudaStream_t) {
  return remap_core(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      0,
      0,
      remapW,
      remapH,
      false,
      true,
      this_image_index,
      dest_image_map,
      true,
      adjustment);
}

template <typename T>
cudaError_t batched_remap_hard_seam_kernel_n(
    const CudaSurface<T>* d_inputs,
    const unsigned short* const* d_mapX_ptrs,
    const unsigned short* const* d_mapY_ptrs,
    const int2* d_offsets,
    const int2* d_sizes,
    int n_images,
    const unsigned char* dest_image_map,
    CudaSurface<T> dest,
    int batchSize,
    cudaStream_t) {
  if (!d_inputs || !d_mapX_ptrs || !d_mapY_ptrs || !d_offsets || !d_sizes || !dest_image_map || !dest.d_ptr) {
    return cudaErrorInvalidDevicePointer;
  }

  for (int y = 0; y < static_cast<int>(dest.height); ++y) {
    for (int x = 0; x < static_cast<int>(dest.width); ++x) {
      const int idx = static_cast<int>(dest_image_map[y * static_cast<int>(dest.width) + x]);
      if (idx < 0 || idx >= n_images) {
        continue;
      }
      const int local_x = x - d_offsets[idx].x;
      const int local_y = y - d_offsets[idx].y;
      if (local_x < 0 || local_y < 0 || local_x >= d_sizes[idx].x || local_y >= d_sizes[idx].y) {
        continue;
      }

      const unsigned short sx = d_mapX_ptrs[idx][local_y * d_sizes[idx].x + local_x];
      const unsigned short sy = d_mapY_ptrs[idx][local_y * d_sizes[idx].x + local_x];
      const CudaSurface<T>& src = d_inputs[idx];
      if (sx >= src.width || sy >= src.height) {
        continue;
      }

      for (int b = 0; b < batchSize; ++b) {
        *surface_ptr_mut(dest, b, x, y) = *surface_ptr_const(src, b, static_cast<int>(sx), static_cast<int>(sy));
      }
    }
  }

  return cudaSuccess;
}

// -----------------------------------------------------------------------------
// Blend 2/3/N
// -----------------------------------------------------------------------------

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendWithContext(
    const T* d_image1,
    const T* d_image2,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext<T>& context,
    int channels,
    cudaStream_t) {
  if (!d_image1 || !d_image2 || !d_mask || !d_output) {
    return cudaErrorInvalidDevicePointer;
  }
  init_level_dims(context.widths, context.heights, context.imageWidth, context.imageHeight);
  context.initialized = true;
  blend_two_images(d_image1, d_image2, d_mask, d_output, context.imageWidth, context.imageHeight, channels, context.batchSize);
  return cudaSuccess;
}

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlend(
    const T* h_image1,
    const T* h_image2,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int numLevels,
    int batchSize,
    cudaStream_t stream) {
  CudaBatchLaplacianBlendContext<T> context(imageWidth, imageHeight, numLevels, batchSize);
  return cudaBatchedLaplacianBlendWithContext<T, F_T>(
      h_image1, h_image2, h_mask, h_output, context, channels, stream);
}

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendWithContext3(
    const T* d_image1,
    const T* d_image2,
    const T* d_image3,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext3<T>& context,
    int channels,
    cudaStream_t) {
  if (!d_image1 || !d_image2 || !d_image3 || !d_mask || !d_output) {
    return cudaErrorInvalidDevicePointer;
  }
  init_level_dims(context.widths, context.heights, context.imageWidth, context.imageHeight);
  context.initialized = true;
  blend_three_images(
      d_image1,
      d_image2,
      d_image3,
      d_mask,
      d_output,
      context.imageWidth,
      context.imageHeight,
      channels,
      context.batchSize);
  return cudaSuccess;
}

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlend3(
    const T* h_image1,
    const T* h_image2,
    const T* h_image3,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int channels,
    int maxLevels,
    int batchSize,
    cudaStream_t stream) {
  CudaBatchLaplacianBlendContext3<T> context(imageWidth, imageHeight, maxLevels, batchSize);
  return cudaBatchedLaplacianBlendWithContext3<T, F_T>(
      h_image1, h_image2, h_image3, h_mask, h_output, context, channels, stream);
}

template <typename T, typename F_T>
cudaError_t cudaBatchedLaplacianBlendOptimized3(
    const T* d_image1,
    const T* d_image2,
    const T* d_image3,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContext3<T>& context,
    int channels,
    cudaStream_t stream) {
  return cudaBatchedLaplacianBlendWithContext3<T, F_T>(
      d_image1, d_image2, d_image3, d_mask, d_output, context, channels, stream);
}

template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendWithContextN(
    const std::vector<const T*>& d_imagePtrs,
    const T* d_mask,
    T* d_output,
    CudaBatchLaplacianBlendContextN<T, N_IMAGES>& context,
    cudaStream_t) {
  if (d_imagePtrs.size() != static_cast<std::size_t>(N_IMAGES) || !d_mask || !d_output) {
    return cudaErrorInvalidValue;
  }
  for (const T* ptr : d_imagePtrs) {
    if (!ptr) {
      return cudaErrorInvalidDevicePointer;
    }
  }
  init_level_dims(context.widths, context.heights, context.imageWidth, context.imageHeight);
  context.initialized = true;
  blend_n_images<T, N_IMAGES, CHANNELS>(d_imagePtrs, d_mask, d_output, context.imageWidth, context.imageHeight, context.batchSize);
  return cudaSuccess;
}

template <typename T, typename F_T, int N_IMAGES, int CHANNELS>
cudaError_t cudaBatchedLaplacianBlendN(
    const std::vector<const T*>& h_imagePtrs,
    const T* h_mask,
    T* h_output,
    int imageWidth,
    int imageHeight,
    int maxLevels,
    int batchSize,
    cudaStream_t stream) {
  CudaBatchLaplacianBlendContextN<T, N_IMAGES> context(imageWidth, imageHeight, maxLevels, batchSize);
  return cudaBatchedLaplacianBlendWithContextN<T, F_T, N_IMAGES, CHANNELS>(
      h_imagePtrs, h_mask, h_output, context, stream);
}

// -----------------------------------------------------------------------------
// 3-image fused helpers
// -----------------------------------------------------------------------------

namespace hm {
namespace pano {
namespace cuda {

template <typename T_pipeline, typename T_compute>
CudaStatus launchFusedRemapToFullKernel3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    CudaMat<T_compute>& cudaFull0,
    CudaMat<T_compute>& cudaFull1,
    CudaMat<T_compute>& cudaFull2,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream) {
  const T_pipeline dflt{};
  const auto remap_one = [&](const CudaMat<T_pipeline>& input,
                             const CudaMat<uint16_t>& mx,
                             const CudaMat<uint16_t>& my,
                             CudaMat<T_compute>& out,
                             int ox,
                             int oy,
                             float3 adj) -> cudaError_t {
    if (apply_adjustment) {
      return batched_remap_kernel_ex_offset_adjust(
          input.surface(), out.surface(), mx.data(), my.data(), dflt, input.batch_size(), mx.width(), mx.height(), ox, oy, false, adj, stream);
    }
    return batched_remap_kernel_ex_offset(
        input.surface(), out.surface(), mx.data(), my.data(), dflt, input.batch_size(), mx.width(), mx.height(), ox, oy, false, stream);
  };

  cudaError_t cuerr = remap_one(
      inputImage0,
      remap_0_x,
      remap_0_y,
      cudaFull0,
      canvas_manager.canvas_positions()[0].x,
      canvas_manager.canvas_positions()[0].y,
      adjustment0);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  cuerr = remap_one(
      inputImage1,
      remap_1_x,
      remap_1_y,
      cudaFull1,
      canvas_manager.canvas_positions()[1].x,
      canvas_manager.canvas_positions()[1].y,
      adjustment1);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  cuerr = remap_one(
      inputImage2,
      remap_2_x,
      remap_2_y,
      cudaFull2,
      canvas_manager.canvas_positions()[2].x,
      canvas_manager.canvas_positions()[2].y,
      adjustment2);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  return CudaStatus::OkStatus();
}

template <typename T_pipeline>
CudaStatus launchFusedRemapHardSeam3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    const CudaMat<unsigned char>& hardSeamMask,
    CudaMat<T_pipeline>& canvas,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream) {
  const T_pipeline dflt{};
  const auto remap_one = [&](const CudaMat<T_pipeline>& input,
                             const CudaMat<uint16_t>& mx,
                             const CudaMat<uint16_t>& my,
                             int image_index,
                             int ox,
                             int oy,
                             float3 adj) -> cudaError_t {
    if (apply_adjustment) {
      return batched_remap_kernel_ex_offset_with_dest_map_adjust(
          input.surface(),
          canvas.surface(),
          mx.data(),
          my.data(),
          dflt,
          image_index,
          hardSeamMask.data(),
          input.batch_size(),
          mx.width(),
          mx.height(),
          ox,
          oy,
          adj,
          stream);
    }
    return batched_remap_kernel_ex_offset_with_dest_map(
        input.surface(),
        canvas.surface(),
        mx.data(),
        my.data(),
        dflt,
        image_index,
        hardSeamMask.data(),
        input.batch_size(),
        mx.width(),
        mx.height(),
        ox,
        oy,
        stream);
  };

  cudaError_t cuerr = remap_one(
      inputImage0,
      remap_0_x,
      remap_0_y,
      0,
      canvas_manager.canvas_positions()[0].x,
      canvas_manager.canvas_positions()[0].y,
      adjustment0);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  cuerr = remap_one(
      inputImage1,
      remap_1_x,
      remap_1_y,
      1,
      canvas_manager.canvas_positions()[1].x,
      canvas_manager.canvas_positions()[1].y,
      adjustment1);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  cuerr = remap_one(
      inputImage2,
      remap_2_x,
      remap_2_y,
      2,
      canvas_manager.canvas_positions()[2].x,
      canvas_manager.canvas_positions()[2].y,
      adjustment2);
  if (cuerr != cudaSuccess) return CudaStatus(cuerr);

  return CudaStatus::OkStatus();
}

} // namespace cuda
} // namespace pano
} // namespace hm

// -----------------------------------------------------------------------------
// Explicit instantiations
// -----------------------------------------------------------------------------

template void adjustImageCudaBatch<uchar3>(uchar3*, int, int, int, const float3&);
template void adjustImageCudaBatch<uchar4>(uchar4*, int, int, int, const float3&);
template void adjustImageCudaBatch<float3>(float3*, int, int, int, const float3&);
template void adjustImageCudaBatch<float4>(float4*, int, int, int, const float3&);

template cudaError_t AlphaConditionalCopy<uchar4>(CudaSurface<uchar4>&, CudaSurface<uchar4>&, int, cudaStream_t);
template cudaError_t AlphaConditionalCopy<float4>(CudaSurface<float4>&, CudaSurface<float4>&, int, cudaStream_t);

template cudaError_t simple_make_full_batch<uchar3, float3>(
    const CudaSurface<uchar3>&, int, int, int, int, int, int, bool, int, CudaSurface<float3>, cudaStream_t);
template cudaError_t simple_make_full_batch<uchar4, float4>(
    const CudaSurface<uchar4>&, int, int, int, int, int, int, bool, int, CudaSurface<float4>, cudaStream_t);
template cudaError_t simple_make_full_batch<float3, float3>(
    const CudaSurface<float3>&, int, int, int, int, int, int, bool, int, CudaSurface<float3>, cudaStream_t);
template cudaError_t simple_make_full_batch<float4, float4>(
    const CudaSurface<float4>&, int, int, int, int, int, int, bool, int, CudaSurface<float4>, cudaStream_t);
template cudaError_t simple_make_full_batch<float3, float4>(
    const CudaSurface<float3>&, int, int, int, int, int, int, bool, int, CudaSurface<float4>, cudaStream_t);
template cudaError_t simple_make_full_batch<float4, float3>(
    const CudaSurface<float4>&, int, int, int, int, int, int, bool, int, CudaSurface<float3>, cudaStream_t);
template cudaError_t simple_make_full_batch<uchar4, float3>(
    const CudaSurface<uchar4>&, int, int, int, int, int, int, bool, int, CudaSurface<float3>, cudaStream_t);
template cudaError_t simple_make_full_batch<uchar3, float4>(
    const CudaSurface<uchar3>&, int, int, int, int, int, int, bool, int, CudaSurface<float4>, cudaStream_t);

template cudaError_t copy_roi_batched<float3, float3>(
    const CudaSurface<float3>&, int, int, int, int, CudaSurface<float3>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<float4, float4>(
    const CudaSurface<float4>&, int, int, int, int, CudaSurface<float4>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<float3, uchar3>(
    const CudaSurface<float3>&, int, int, int, int, CudaSurface<uchar3>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<float4, uchar4>(
    const CudaSurface<float4>&, int, int, int, int, CudaSurface<uchar4>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<float4, uchar3>(
    const CudaSurface<float4>&, int, int, int, int, CudaSurface<uchar3>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<float3, uchar4>(
    const CudaSurface<float3>&, int, int, int, int, CudaSurface<uchar4>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<uchar3, uchar3>(
    const CudaSurface<uchar3>&, int, int, int, int, CudaSurface<uchar3>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<uchar4, uchar4>(
    const CudaSurface<uchar4>&, int, int, int, int, CudaSurface<uchar4>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<uchar3, float3>(
    const CudaSurface<uchar3>&, int, int, int, int, CudaSurface<float3>, int, int, int, cudaStream_t);
template cudaError_t copy_roi_batched<uchar4, float4>(
    const CudaSurface<uchar4>&, int, int, int, int, CudaSurface<float4>, int, int, int, cudaStream_t);

template cudaError_t batched_remap_kernel_ex<float3, float3>(
    const float3*, int, int, float3*, int, int, const unsigned short*, const unsigned short*, float3, int, cudaStream_t);
template cudaError_t batched_remap_kernel_ex<float4, float4>(
    const float4*, int, int, float4*, int, int, const unsigned short*, const unsigned short*, float4, int, cudaStream_t);

template cudaError_t batched_remap_kernel_ex_offset<float3, float3>(
    const CudaSurface<float3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    float3,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<uchar3, float3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<uchar3, uchar3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<uchar3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<float1, float1>(
    const CudaSurface<float1>&,
    const CudaSurface<float1>&,
    const unsigned short*,
    const unsigned short*,
    float1,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<float4, float4>(
    const CudaSurface<float4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    float4,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<uchar4, float4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset<uchar4, uchar4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<uchar4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);

template cudaError_t batched_remap_kernel_ex_offset_roi<float3, float3>(
    const CudaSurface<float3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    float3,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<uchar3, float3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<uchar3, uchar3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<uchar3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<float1, float1>(
    const CudaSurface<float1>&,
    const CudaSurface<float1>&,
    const unsigned short*,
    const unsigned short*,
    float1,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<float4, float4>(
    const CudaSurface<float4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    float4,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<uchar4, float4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_roi<uchar4, uchar4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<uchar4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    int,
    bool,
    cudaStream_t);

template cudaError_t batched_remap_kernel_ex_offset_adjust<float3, float3>(
    const CudaSurface<float3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    float3,
    int,
    int,
    int,
    int,
    int,
    bool,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_adjust<uchar3, uchar3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<uchar3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    int,
    int,
    int,
    int,
    bool,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_adjust<float4, float4>(
    const CudaSurface<float4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    float4,
    int,
    int,
    int,
    int,
    int,
    bool,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_adjust<uchar4, float4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    bool,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_adjust<uchar4, uchar4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<uchar4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    int,
    int,
    int,
    int,
    bool,
    float3,
    cudaStream_t);

template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<float1, float1>(
    const CudaSurface<float1>&,
    const CudaSurface<float1>&,
    const unsigned short*,
    const unsigned short*,
    float1,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<float3, float3>(
    const CudaSurface<float3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    float3,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<uchar1, uchar1>(
    const CudaSurface<uchar1>&,
    const CudaSurface<uchar1>&,
    const unsigned short*,
    const unsigned short*,
    uchar1,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<uchar3, uchar3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<uchar3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<float4, float4>(
    const CudaSurface<float4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    float4,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<uchar4, uchar4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<uchar4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);

template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<float3, float3>(
    const CudaSurface<float3>&,
    const CudaSurface<float3>&,
    const unsigned short*,
    const unsigned short*,
    float3,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<uchar3, uchar3>(
    const CudaSurface<uchar3>&,
    const CudaSurface<uchar3>&,
    const unsigned short*,
    const unsigned short*,
    uchar3,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<float4, float4>(
    const CudaSurface<float4>&,
    const CudaSurface<float4>&,
    const unsigned short*,
    const unsigned short*,
    float4,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    float3,
    cudaStream_t);
template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<uchar4, uchar4>(
    const CudaSurface<uchar4>&,
    const CudaSurface<uchar4>&,
    const unsigned short*,
    const unsigned short*,
    uchar4,
    int,
    const unsigned char*,
    int,
    int,
    int,
    int,
    int,
    float3,
    cudaStream_t);

template cudaError_t batched_remap_hard_seam_kernel_n<uchar3>(
    const CudaSurface<uchar3>*, const unsigned short* const*, const unsigned short* const*, const int2*, const int2*, int, const unsigned char*, CudaSurface<uchar3>, int, cudaStream_t);
template cudaError_t batched_remap_hard_seam_kernel_n<uchar4>(
    const CudaSurface<uchar4>*, const unsigned short* const*, const unsigned short* const*, const int2*, const int2*, int, const unsigned char*, CudaSurface<uchar4>, int, cudaStream_t);
template cudaError_t batched_remap_hard_seam_kernel_n<float3>(
    const CudaSurface<float3>*, const unsigned short* const*, const unsigned short* const*, const int2*, const int2*, int, const unsigned char*, CudaSurface<float3>, int, cudaStream_t);
template cudaError_t batched_remap_hard_seam_kernel_n<float4>(
    const CudaSurface<float4>*, const unsigned short* const*, const unsigned short* const*, const int2*, const int2*, int, const unsigned char*, CudaSurface<float4>, int, cudaStream_t);

template cudaError_t cudaBatchedLaplacianBlend<float, float>(
    const float*, const float*, const float*, float*, int, int, int, int, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlend<unsigned char, float>(
    const unsigned char*, const unsigned char*, const unsigned char*, unsigned char*, int, int, int, int, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContext<float, float>(
    const float*, const float*, const float*, float*, CudaBatchLaplacianBlendContext<float>&, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContext<unsigned char, float>(
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    unsigned char*,
    CudaBatchLaplacianBlendContext<unsigned char>&,
    int,
    cudaStream_t);

template cudaError_t cudaBatchedLaplacianBlend3<float, float>(
    const float*, const float*, const float*, const float*, float*, int, int, int, int, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlend3<unsigned char, float>(
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    unsigned char*,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContext3<float, float>(
    const float*, const float*, const float*, const float*, float*, CudaBatchLaplacianBlendContext3<float>&, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendWithContext3<unsigned char, float>(
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    unsigned char*,
    CudaBatchLaplacianBlendContext3<unsigned char>&,
    int,
    cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendOptimized3<float, float>(
    const float*, const float*, const float*, const float*, float*, CudaBatchLaplacianBlendContext3<float>&, int, cudaStream_t);
template cudaError_t cudaBatchedLaplacianBlendOptimized3<unsigned char, float>(
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    const unsigned char*,
    unsigned char*,
    CudaBatchLaplacianBlendContext3<unsigned char>&,
    int,
    cudaStream_t);

#define INSTANTIATE_BLEND_N(N, C)                                                                                 \
  template cudaError_t cudaBatchedLaplacianBlendWithContextN<float, float, N, C>(                                \
      const std::vector<const float*>&, const float*, float*, CudaBatchLaplacianBlendContextN<float, N>&, cudaStream_t); \
  template cudaError_t cudaBatchedLaplacianBlendN<float, float, N, C>(                                            \
      const std::vector<const float*>&, const float*, float*, int, int, int, int, cudaStream_t);

INSTANTIATE_BLEND_N(2, 3)
INSTANTIATE_BLEND_N(3, 3)
INSTANTIATE_BLEND_N(4, 3)
INSTANTIATE_BLEND_N(5, 3)
INSTANTIATE_BLEND_N(6, 3)
INSTANTIATE_BLEND_N(7, 3)
INSTANTIATE_BLEND_N(8, 3)
INSTANTIATE_BLEND_N(2, 4)
INSTANTIATE_BLEND_N(3, 4)
INSTANTIATE_BLEND_N(4, 4)
INSTANTIATE_BLEND_N(5, 4)
INSTANTIATE_BLEND_N(6, 4)
INSTANTIATE_BLEND_N(7, 4)
INSTANTIATE_BLEND_N(8, 4)

#undef INSTANTIATE_BLEND_N

namespace hm {
namespace pano {
namespace cuda {

template CudaStatus launchFusedRemapToFullKernel3<uchar3, float3>(
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    CudaMat<float3>&,
    CudaMat<float3>&,
    CudaMat<float3>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapToFullKernel3<uchar3, float4>(
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapToFullKernel3<uchar4, float4>(
    const CudaMat<uchar4>&,
    const CudaMat<uchar4>&,
    const CudaMat<uchar4>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapToFullKernel3<float3, float3>(
    const CudaMat<float3>&,
    const CudaMat<float3>&,
    const CudaMat<float3>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    CudaMat<float3>&,
    CudaMat<float3>&,
    CudaMat<float3>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapToFullKernel3<float4, float4>(
    const CudaMat<float4>&,
    const CudaMat<float4>&,
    const CudaMat<float4>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    CudaMat<float4>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapHardSeam3<uchar3>(
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uchar3>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<unsigned char>&,
    CudaMat<uchar3>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapHardSeam3<uchar4>(
    const CudaMat<uchar4>&,
    const CudaMat<uchar4>&,
    const CudaMat<uchar4>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<unsigned char>&,
    CudaMat<uchar4>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapHardSeam3<float3>(
    const CudaMat<float3>&,
    const CudaMat<float3>&,
    const CudaMat<float3>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<unsigned char>&,
    CudaMat<float3>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

template CudaStatus launchFusedRemapHardSeam3<float4>(
    const CudaMat<float4>&,
    const CudaMat<float4>&,
    const CudaMat<float4>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<uint16_t>&,
    const CudaMat<unsigned char>&,
    CudaMat<float4>&,
    const CanvasManager3&,
    float3,
    float3,
    float3,
    bool,
    cudaStream_t);

} // namespace cuda
} // namespace pano
} // namespace hm
