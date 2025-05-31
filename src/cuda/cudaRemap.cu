#include <cuda_runtime.h>
// #if (CUDART_VERSION >= 11000)
// #include <cuda_bf16.h>
// #endif
#include <cuda_fp16.h>
#include <device_launch_parameters.h>
#include "cudaImageAdjust.h"
#include "cudaRemap.h" // Assumed to declare these host functions

#include <cstdint>
#include <limits>
#include <iostream>

namespace {

constexpr unsigned short kUnmappedPositionValue = std::numeric_limits<unsigned short>::max();

template <typename T_in, typename T_out>
inline T_out __device__ cast_to(const T_in& in) {
  return static_cast<T_out>(in);
}

template <>
inline float3 __device__ cast_to(const uchar3& in) {
  return float3{
      .x = static_cast<float>(in.x),
      .y = static_cast<float>(in.y),
      .z = static_cast<float>(in.z),
  };
}

template <>
inline float4 __device__ cast_to(const uchar4& in) {
  return float4{
      .x = static_cast<float>(in.x),
      .y = static_cast<float>(in.y),
      .z = static_cast<float>(in.z),
      .w = static_cast<float>(in.w)};
}

//------------------------------------------------------------------------------
// Templated Batched Remap Kernel EX (unchanged)
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelEx(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
    const unsigned short* mapX,
    const unsigned short* mapY,
    T_in deflt,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destW || y >= destH)
    return;

  int srcImageSize = srcW * srcH;
  int destImageSize = destW * destH;

  T_out* destImage = dest + b * destImageSize;

  const int destIdx = y * destW + x;
  const int srcX = static_cast<int>(mapX[destIdx]);
  const int srcY = static_cast<int>(mapY[destIdx]);

  if (srcX < srcW && srcY < srcH) {
    const T_in* srcImage = src + b * srcImageSize;
    int srcIdx = srcY * srcW + srcX;
    destImage[destIdx] = static_cast<T_out>(srcImage[srcIdx]);
  } else {
    destImage[destIdx] = deflt;
  }
}

//------------------------------------------------------------------------------
// NEW: Templated Batched Remap Kernel EX with Offset (Single-channel)
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffset(
    const CudaSurface<T_in> src,
    CudaSurface<T_out> dest,
    const unsigned short* mapX, // mapping arrays of size (remapW x remapH)
    const unsigned short* mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= dest.width || destY < 0 || destY >= dest.height)
    return;

  int destImageSizeBytes = dest.height * dest.pitch;

  T_out* destImage = advance_bytes(dest.d_ptr, b * destImageSizeBytes);
  T_out* dest_pos = advance_bytes(destImage, destY * dest.pitch) + destX;

  int mapIdx = y * remapW + x;

  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if (srcX < src.width && srcY < src.height) {
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    *dest_pos = cast_to<T_in, T_out>(*src_pos);
  } else {
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      *dest_pos = cast_to<T_in, T_out>(deflt);
      if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
        // Has an alpha channel, so clear it
        if (srcX == kUnmappedPositionValue) {
          dest_pos->w = 0;
        }
      }
    }
  }
}

template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetAdjust(
    const CudaSurface<T_in> src,
    CudaSurface<T_out> dest,
    const unsigned short* mapX, // mapping arrays of size (remapW x remapH)
    const unsigned short* mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    bool no_unmapped_write,
    float3 adjustment) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= dest.width || destY < 0 || destY >= dest.height)
    return;

  int destImageSizeBytes = dest.height * dest.pitch;

  T_out* destImage = advance_bytes(dest.d_ptr, b * destImageSizeBytes);
  T_out* dest_pos = advance_bytes(destImage, destY * dest.pitch) + destX;

  int mapIdx = y * remapW + x;

  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if (srcX < src.width && srcY < src.height) {
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    *dest_pos = PixelAdjuster<T_out>::adjust(static_cast<T_out>(*src_pos), adjustment);
  } else {
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      *dest_pos = cast_to<T_in, T_out>(deflt);
      if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
        // Has an alpha channel, so clear it
        if (srcX == kUnmappedPositionValue) {
          dest_pos->w = 0;
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// NEW: Templated Batched Remap Kernel EX with Offset (Single-channel)
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetWithDestMap(
    const CudaSurface<T_in> src,
    CudaSurface<T_out> dest,
    const unsigned short* mapX, // mapping arrays of size (remapW x remapH)
    const unsigned short* mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= dest.width || destY < 0 || destY >= dest.height)
    return;

  // No pitch on the mask right now
  int checkIdx = (offsetY + y) * dest.width + (offsetX + x);
  // printf("dest_image_map[checkIdx]=%d\n", (int)dest_image_map[checkIdx]);
  if (dest_image_map[checkIdx] == this_image_index) {
    int destImageSizeBytes = dest.width * dest.pitch;
    T_out* destImage = advance_bytes(dest.d_ptr, b * destImageSizeBytes);
    T_out* dest_pos = advance_bytes(destImage, destY * dest.pitch) + destX;

    int mapIdx = y * remapW + x;
    int srcX = static_cast<int>(mapX[mapIdx]);
    int srcY = static_cast<int>(mapY[mapIdx]);

    if (srcX < src.width && srcY < src.height) {
      int srcImageSizeBytes = src.width * src.pitch;
      const T_in* src_pos = advance_bytes(src.d_ptr, srcImageSizeBytes * b + srcY * src.pitch) + srcX;
      *dest_pos = static_cast<T_out>(*src_pos);
    } else {
      *dest_pos = deflt;
    }
  }
}

template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetWithDestMapAdjust(
    const CudaSurface<T_in> src,
    CudaSurface<T_out> dest,
    const unsigned short* mapX, // mapping arrays of size (remapW x remapH)
    const unsigned short* mapY,
    T_in deflt,
    int this_image_index,
    const unsigned char* dest_image_map,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    float3 adjustment) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= dest.width || destY < 0 || destY >= dest.height)
    return;

  int checkIdx = (offsetY + y) * dest.width + (offsetX + x);

  if (dest_image_map[checkIdx] == this_image_index) {
    int mapIdx = y * remapW + x;
    int srcX = static_cast<int>(mapX[mapIdx]);
    int srcY = static_cast<int>(mapY[mapIdx]);
    if (srcX < src.width && srcY < src.height) {
      // Out is more likely to be a float, so adjust after any cast
      *surface_ptr(dest, b, destX, destY) = PixelAdjuster<T_out>::adjust(*surface_ptr(src, b, srcX, srcY), adjustment);
    } else {
      // destImage[destIdx] = deflt;
      *surface_ptr(dest, b, destX, destY) = deflt;
    }
  }
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Host Function: Batched Remap EX (unchanged)
//------------------------------------------------------------------------------
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
    T_in deflt,
    int batchSize,
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelEx<T_in, T_out>
      <<<gridDim, blockDim, 0, stream>>>(d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, deflt, batchSize);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// Host Function: Batched Remap EX with Offset (Single-channel)
//------------------------------------------------------------------------------

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
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffset<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      src, dest, d_mapX, d_mapY, deflt, batchSize, remapW, remapH, offsetX, offsetY, no_unmapped_write);
  return cudaGetLastError();
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
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffsetAdjust<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      src, dest, d_mapX, d_mapY, deflt, batchSize, remapW, remapH, offsetX, offsetY, no_unmapped_write, adjustment);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// NEW: Host Function: Batched Remap EX with Offset (Single-channel)
//------------------------------------------------------------------------------
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
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffsetWithDestMap<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      src, dest, d_mapX, d_mapY, deflt, this_image_index, dest_image_map, batchSize, remapW, remapH, offsetX, offsetY);
  return cudaGetLastError();
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
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffsetWithDestMapAdjust<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      this_image_index,
      dest_image_map,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      adjustment);
  return cudaGetLastError();
}

// Macro for instantiating batched_remap_kernel_ex_offset<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_offset<T_in, T_out>( \
      const CudaSurface<T_in>& src,                                 \
      const CudaSurface<T_out>& dest,                               \
      const unsigned short* d_mapX,                                 \
      const unsigned short* d_mapY,                                 \
      T_in deflt,                                                   \
      int batchSize,                                                \
      int remapW,                                                   \
      int remapH,                                                   \
      int offsetX,                                                  \
      int offsetY,                                                  \
      bool no_unmapped_write,                                       \
      cudaStream_t stream);

#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_offset_adjust<T_in, T_out>( \
      const CudaSurface<T_in>& src,                                        \
      const CudaSurface<T_out>& dest,                                      \
      const unsigned short* d_mapX,                                        \
      const unsigned short* d_mapY,                                        \
      T_in deflt,                                                          \
      int batchSize,                                                       \
      int remapW,                                                          \
      int remapH,                                                          \
      int offsetX,                                                         \
      int offsetY,                                                         \
      bool no_unmapped_write,                                              \
      float3 adjustment,                                                   \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex<T_in, T_out>( \
      const T_in* d_src,                                     \
      int srcW,                                              \
      int srcH,                                              \
      T_out* d_dest,                                         \
      int destW,                                             \
      int destH,                                             \
      const unsigned short* d_mapX,                          \
      const unsigned short* d_mapY,                          \
      T_in deflt,                                            \
      int batchSize,                                         \
      cudaStream_t stream);

#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_ADJUST(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_adjust<T_in, T_out>( \
      const T_in* d_src,                                            \
      int srcW,                                                     \
      int srcH,                                                     \
      T_out* d_dest,                                                \
      int destW,                                                    \
      int destH,                                                    \
      const unsigned short* d_mapX,                                 \
      const unsigned short* d_mapY,                                 \
      T_in deflt,                                                   \
      int batchSize,                                                \
      float3 adjustment cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex_offset_with_dest_map<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<T_in, T_out>( \
      const CudaSurface<T_in>& src,                                               \
      const CudaSurface<T_out>& dest,                                             \
      const unsigned short* d_mapX,                                               \
      const unsigned short* d_mapY,                                               \
      T_in deflt,                                                                 \
      int this_image_index,                                                       \
      const unsigned char* dest_image_map,                                        \
      int batchSize,                                                              \
      int remapW,                                                                 \
      int remapH,                                                                 \
      int offsetX,                                                                \
      int offsetY,                                                                \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex_offset_with_dest_map_adjust<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<T_in, T_out>( \
      const CudaSurface<T_in>& src,                                                      \
      const CudaSurface<T_out>& dest,                                                    \
      const unsigned short* d_mapX,                                                      \
      const unsigned short* d_mapY,                                                      \
      T_in deflt,                                                                        \
      int this_image_index,                                                              \
      const unsigned char* dest_image_map,                                               \
      int batchSize,                                                                     \
      int remapW,                                                                        \
      int remapH,                                                                        \
      int offsetX,                                                                       \
      int offsetY,                                                                       \
      float3 adjustment,                                                                 \
      cudaStream_t stream);

// For batched_remap_kernel_ex_offset
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(float, float)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar4, uchar4)
// INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(__half, __half)

INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(uchar4, uchar4)

// For batched_remap_kernel_ex
INSTANTIATE_BATCHED_REMAP_KERNEL_EX(float3, float3)

// Instantiate for float input and float output:
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(float, float)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(uchar1, uchar1)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(uchar4, uchar4)

INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(uchar4, uchar4)

// Instantiate for __half input and __half output:
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(__half, __half)
