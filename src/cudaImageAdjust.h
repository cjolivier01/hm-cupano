#pragma once

#include <cuda_runtime.h>

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
struct PixelAdjuster {
  __device__ static T adjust(const T& pixel, const float3& adjustment);
};

