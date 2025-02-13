#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

/*----------------------------------------------------------------------------
  Additional vector types for half (float16), USHORT, and bfloat16.

  CUDA provides __half and __half4 in cuda_fp16.h, but we want to use our own custom half4.
  CUDA’s built-in vector types (e.g. uchar3) are available for USHORT as well.
-----------------------------------------------------------------------------*/

#ifndef HALF3_DEFINED
#define HALF3_DEFINED
/**
 * @brief 3-element vector of __half values.
 */
struct half3 {
  __half x, y, z;
};
#endif

#ifndef HALF4_DEFINED
#define HALF4_DEFINED
/**
 * @brief Custom 4-element vector of __half values.
 *
 * This struct is defined to replace CUDA’s __half4.
 */
struct half4 {
  __half x, y, z, w;
};
#endif

// USHORT types are typically provided by CUDA as "ushort3", "ushort4" etc.
// If not available, you could define your own similar to half3/half4.
#ifndef BF16_3_DEFINED
#define BF16_3_DEFINED
/**
 * @brief 3-element vector of __nv_bfloat16 values.
 */
struct bfloat16_3 {
  __nv_bfloat16 x, y, z;
};
#endif

#ifndef BF16_4_DEFINED
#define BF16_4_DEFINED
/**
 * @brief 4-element vector of __nv_bfloat16 values.
 */
struct bfloat16_4 {
  __nv_bfloat16 x, y, z, w;
};
#endif

//------------------------------------------------------------------------------
// Trait templates to convert a pointer to a CUDA vector type into a pointer
// to its base scalar type.
//------------------------------------------------------------------------------

/**
 * @brief Primary template for BaseScalar.
 *
 * This template should be specialized for each CUDA vector or scalar type.
 */
template <typename T>
struct BaseScalar; // no definition for the primary template

// --- 1-channel types ---
template <>
struct BaseScalar<unsigned char> {
  using type = unsigned char;
};

template <>
struct BaseScalar<uchar1> {
  using type = unsigned char;
};

template <>
struct BaseScalar<unsigned short> {
  using type = unsigned short;
};

template <>
struct BaseScalar<int> {
  using type = int;
};

template <>
struct BaseScalar<float> {
  using type = float;
};

template <>
struct BaseScalar<__half> {
  using type = __half;
};

template <>
struct BaseScalar<__nv_bfloat16> {
  using type = __nv_bfloat16;
};

// --- 3-channel types ---
template <>
struct BaseScalar<uchar3> {
  using type = unsigned char;
};

template <>
struct BaseScalar<ushort3> {
  using type = unsigned short;
};

template <>
struct BaseScalar<int3> {
  using type = int;
};

template <>
struct BaseScalar<float3> {
  using type = float;
};

template <>
struct BaseScalar<half3> {
  using type = __half;
};

template <>
struct BaseScalar<bfloat16_3> {
  using type = __nv_bfloat16;
};

// --- 4-channel types ---
template <>
struct BaseScalar<uchar4> {
  using type = unsigned char;
};

template <>
struct BaseScalar<ushort4> {
  using type = unsigned short;
};

template <>
struct BaseScalar<int4> {
  using type = int;
};

template <>
struct BaseScalar<float4> {
  using type = float;
};

template <>
struct BaseScalar<half4> {
  using type = __half;
};

template <>
struct BaseScalar<bfloat16_4> {
  using type = __nv_bfloat16;
};

/**
 * @brief Helper alias to obtain the base scalar type.
 *
 * For example, BaseScalar_t<float3> is float, and BaseScalar_t<half4> is __half.
 */
template <typename T>
using BaseScalar_t = typename BaseScalar<T>::type;

template <typename T>
struct CudaSurface {
  T* d_ptr;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t pitch;
};
