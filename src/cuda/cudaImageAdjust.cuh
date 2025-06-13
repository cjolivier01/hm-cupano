#pragma once

#include <cuda_runtime.h>

template <typename T>
void adjustImageCudaBatch(T* d_image, int batchSize, int width, int height, const float3& adjustment);

#ifdef __CUDACC__
template <typename T>
struct PixelAdjuster {
  __device__ static T adjust(const T& pixel, const float3& adjustment);
};

//------------------------------------------------------------------------------
// Device helper: clampf
//
// Clamps a float value between minVal and maxVal.
//------------------------------------------------------------------------------
inline __device__ float clampf(float val, float minVal, float maxVal) {
  return fminf(maxVal, fmaxf(minVal, val));
}

//------------------------------------------------------------------------------
// PixelAdjuster: A traits struct with specializations for different pixel types.
//
// Each specialization provides a static __device__ method "adjust" that applies a
// per-channel adjustment (passed as a float3) to a pixel value. For 8-bit types,
// the result is clamped to the range [0,255]. For floating-point types, the adjustment
// is applied directly (and alpha is preserved in 4-channel types).
//------------------------------------------------------------------------------

// Primary template declaration (undefined).
template <typename T>
struct PixelAdjuster;

// Specialization for uchar3 (3-channel 8-bit pixel)
template <>
struct PixelAdjuster<uchar3> {
  __device__ static uchar3 adjust(const uchar3& pixel, const float3& adjustment) {
    uchar3 out;
    out.x = static_cast<unsigned char>(clampf(float(pixel.x) + adjustment.x, 0.0f, 255.0f));
    out.y = static_cast<unsigned char>(clampf(float(pixel.y) + adjustment.y, 0.0f, 255.0f));
    out.z = static_cast<unsigned char>(clampf(float(pixel.z) + adjustment.z, 0.0f, 255.0f));
    return out;
  }
};

// Specialization for uchar4 (4-channel 8-bit pixel)
template <>
struct PixelAdjuster<uchar4> {
  __device__ static uchar4 adjust(const uchar4& pixel, const float3& adjustment) {
    uchar4 out;
    out.x = static_cast<unsigned char>(clampf(float(pixel.x) + adjustment.x, 0.0f, 255.0f));
    out.y = static_cast<unsigned char>(clampf(float(pixel.y) + adjustment.y, 0.0f, 255.0f));
    out.z = static_cast<unsigned char>(clampf(float(pixel.z) + adjustment.z, 0.0f, 255.0f));
    out.w = pixel.w; // Preserve alpha.
    return out;
  }
};

// Specialization for float3 (3-channel floating-point pixel)
template <>
struct PixelAdjuster<float3> {
  __device__ static float3 adjust(const float3& pixel, const float3& adjustment) {
    float3 out;
    out.x = pixel.x + adjustment.x;
    out.y = pixel.y + adjustment.y;
    out.z = pixel.z + adjustment.z;
    return out;
  }
};

// Specialization for float4 (4-channel floating-point pixel)
template <>
struct PixelAdjuster<float4> {
  __device__ static float4 adjust(const float4& pixel, const float3& adjustment) {
    float4 out;
    out.x = pixel.x + adjustment.x;
    out.y = pixel.y + adjustment.y;
    out.z = pixel.z + adjustment.z;
    out.w = pixel.w; // Preserve alpha.
    return out;
  }
};

#endif
