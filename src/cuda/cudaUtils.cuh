#pragma once

#include <cuda_runtime.h>
#include "cudaTypes.h"

#include <algorithm>
#include <cstdint>

namespace hm {
namespace cupano {
namespace cuda {

template <typename T, typename TLO, typename THI>
__device__ T clamp(const TLO& lo, const T& val, const THI& hi) {
  return val < T(lo) ? T(lo) : (val > T(hi) ? T(hi) : val);
}

template <typename F_dest, typename F, typename COMPUTE_F = float>
__device__ inline unsigned char round_to_uchar(const F& x) {
  COMPUTE_F x_rounded = static_cast<COMPUTE_F>(x) /* + 0.5f*/;
  if (x_rounded <= 0.0f) {
    return 0;
  }
  if (x_rounded >= 255.0f) {
    return 255;
  }
  return static_cast<F_dest>(static_cast<unsigned char>(x_rounded)); // Cast result to unsigned char
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"

template <typename T>
inline __device__ T max_of(const T& v1, const T& v2) {
  if constexpr (std::is_same<T, __half>::value) {
    return (T)std::max((float)v1, (float)v2);
  } else {
    return std::max(v1, v2);
  }
}

template <typename T>
inline __device__ T is_zero(const T& v) {
  if constexpr (std::is_same<T, __half>::value) {
    constexpr unsigned short uszero = 0;
    return static_cast<T>(uszero);
  } else {
    return v == static_cast<T>(0);
  }
}

#pragma GCC diagnostic pop

// template <typename T_in, typename T_out>
// inline T_out __device__ cast_to(const T_in& in) {
//   return static_cast<T_out>(in);
// }

// template <>
// inline float3 __device__ cast_to(const uchar3& in) {
//   return float3{
//       .x = static_cast<float>(in.x),
//       .y = static_cast<float>(in.y),
//       .z = static_cast<float>(in.z),
//   };
// }

// template <>
// inline float4 __device__ cast_to(const uchar4& in) {
//   return float4{
//       .x = static_cast<float>(in.x),
//       .y = static_cast<float>(in.y),
//       .z = static_cast<float>(in.z),
//       .w = static_cast<float>(in.w)};
// }

#define DISALLOW_UNSIGNED_CHAR(_type$) \
  static_assert(!std::is_same<_type$, unsigned char>::value, "Error: unsigned char is not allowed for this function.")

template <typename T_dest, typename T_src>
__device__ inline T_dest perform_cast(const T_src& src) {
  // Make sure this isn't casting anything down to char without proper clamping
  static_assert(sizeof(T_dest) != 1 || sizeof(T_src) <= sizeof(T_dest));
  return static_cast<T_dest>(src);
}

#define DECLARE_PERFORM_CAST_UCHAR_3(_src$)                                                  \
  template <>                                                                                \
  __device__ inline uchar3 perform_cast(const _src$& src) {                                  \
    return uchar3{                                                                           \
        .x = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.x)), \
        .y = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.y)), \
        .z = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.z)), \
    };                                                                                       \
  }

#define DECLARE_PERFORM_CAST_3(_src$, _dest$)               \
  template <>                                               \
  __device__ inline _dest$ perform_cast(const _src$& src) { \
    DISALLOW_UNSIGNED_CHAR(_dest$);                         \
    return _dest${                                          \
        .x = static_cast<BaseScalar_t<_dest$>>(src.x),      \
        .y = static_cast<BaseScalar_t<_dest$>>(src.y),      \
        .z = static_cast<BaseScalar_t<_dest$>>(src.z),      \
    };                                                      \
  }

// #define DECLARE_PERFORM_CAST_H_TO_UCHAR3(_src$)                                              \
//   template <>                                                                                \
//   __device__ inline uchar3 perform_cast(const _src$& src) {                                  \
//     return uchar3{                                                                           \
//         .x = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.x)), \
//         .y = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.y)), \
//         .z = static_cast<BaseScalar_t<uchar3>>(round_to_uchar<BaseScalar_t<uchar3>>(src.z)), \
//     };                                                                                       \
//   }

#define DECLARE_PERFORM_CAST_UCHAR_4(_src$)                                                  \
  template <>                                                                                \
  __device__ inline uchar4 perform_cast(const _src$& src) {                                  \
    return uchar4{                                                                           \
        .x = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.x)), \
        .y = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.y)), \
        .z = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.z)), \
        .w = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.w)), \
    };                                                                                       \
  }

#define DECLARE_PERFORM_CAST_4(_src$, _dest$)               \
  template <>                                               \
  __device__ inline _dest$ perform_cast(const _src$& src) { \
    DISALLOW_UNSIGNED_CHAR(_dest$);                         \
    return _dest${                                          \
        .x = static_cast<BaseScalar_t<_dest$>>(src.x),      \
        .y = static_cast<BaseScalar_t<_dest$>>(src.y),      \
        .z = static_cast<BaseScalar_t<_dest$>>(src.z),      \
        .w = static_cast<BaseScalar_t<_dest$>>(src.w),      \
    };                                                      \
  }

#define DECLARE_PERFORM_CAST_3_TO_4(_src$, _dest$)                                     \
  template <>                                                                          \
  __device__ inline _dest$ perform_cast(const _src$& src) {                            \
    DISALLOW_UNSIGNED_CHAR(_dest$);                                                    \
    return _dest${                                                                     \
        .x = static_cast<BaseScalar_t<_dest$>>(src.x),                                 \
        .y = static_cast<BaseScalar_t<_dest$>>(src.y),                                 \
        .z = static_cast<BaseScalar_t<_dest$>>(src.z),                                 \
        .w = static_cast<BaseScalar_t<_dest$>>(static_cast<BaseScalar_t<_src$>>(255)), \
    };                                                                                 \
  }

#define DECLARE_PERFORM_CAST_F3_TO_UCHAR_4(_src$)                                            \
  template <>                                                                                \
  __device__ inline uchar4 perform_cast(const _src$& src) {                                  \
    return uchar4{                                                                           \
        .x = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.x)), \
        .y = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.y)), \
        .z = static_cast<BaseScalar_t<uchar4>>(round_to_uchar<BaseScalar_t<uchar4>>(src.z)), \
        .w = 255,                                                                            \
    };                                                                                       \
  }

#define DECLARE_PERFORM_CAST_UCHAR4_TO_3(_dest$)             \
  /* We simply discard the fourth channel (alpha) */         \
  template <>                                                \
  __device__ inline _dest$ perform_cast(const uchar4& src) { \
    return _dest${                                           \
        .x = static_cast<BaseScalar_t<_dest$>>(src.x),       \
        .y = static_cast<BaseScalar_t<_dest$>>(src.y),       \
        .z = static_cast<BaseScalar_t<_dest$>>(src.z),       \
    };                                                       \
  }

DECLARE_PERFORM_CAST_UCHAR_3(float4)
DECLARE_PERFORM_CAST_UCHAR_3(float3)
DECLARE_PERFORM_CAST_UCHAR_3(half3)
DECLARE_PERFORM_CAST_UCHAR_3(half4)

DECLARE_PERFORM_CAST_3_TO_4(uchar3, float4)
DECLARE_PERFORM_CAST_3_TO_4(uchar3, half4)
DECLARE_PERFORM_CAST_3_TO_4(float3, half4)

DECLARE_PERFORM_CAST_3(uchar3, float3)
DECLARE_PERFORM_CAST_3(uchar3, half3)

DECLARE_PERFORM_CAST_UCHAR_4(float4)
DECLARE_PERFORM_CAST_UCHAR_4(half4)

DECLARE_PERFORM_CAST_4(uchar4, float4)
DECLARE_PERFORM_CAST_4(uchar4, half4)

DECLARE_PERFORM_CAST_F3_TO_UCHAR_4(float3)
DECLARE_PERFORM_CAST_F3_TO_UCHAR_4(half3)

DECLARE_PERFORM_CAST_UCHAR4_TO_3(float3)
DECLARE_PERFORM_CAST_UCHAR4_TO_3(half3)

} // namespace cuda
} // namespace cupano
} // namespace hm
