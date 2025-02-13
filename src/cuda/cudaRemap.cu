#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaImageAdjust.h"
#include "cudaRemap.h" // Assumed to declare these host functions

#include <cstdint>
#include <limits>

namespace {

constexpr unsigned short kUnmappedPositionValue = std::numeric_limits<unsigned short>::max();

template <typename T_in, typename T_out>
T_out __device__ cast_to(const T_in& in) {
  return static_cast<T_out>(in);
}

template <>
float3 __device__ cast_to(const uchar3& in) {
  return float3{
      .x = static_cast<float>(in.x),
      .y = static_cast<float>(in.y),
      .z = static_cast<float>(in.z),
  };
}

template <>
float4 __device__ cast_to(const uchar4& in) {
  return float4{
      .x = static_cast<float>(in.x),
      .y = static_cast<float>(in.y),
      .z = static_cast<float>(in.z),
      .w = static_cast<float>(in.w)};
}

// template <typename T_in, typename T_out>
// T_out __device__ cast_to(const T_in& in) {
//   return static_cast<T_out>(in);
// }

//------------------------------------------------------------------------------
// Templated Remap Kernel for a Single Image (unchanged)
//------------------------------------------------------------------------------
// template <typename T_in, typename T_out>
// __global__ void remapKernel(
//     const T_in* src,
//     int srcW,
//     int srcH,
//     T_out* dest,
//     int destW,
//     int destH,
//     const unsigned short* mapX,
//     const unsigned short* mapY,
//     T_out defR,
//     T_out defG,
//     T_out defB) {
//   // Compute destination pixel coordinates.
//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x >= destW || y >= destH)
//     return;

//   int destIdx = y * destW + x;

//   // Get mapping coordinates and cast them to int.
//   int srcX = static_cast<int>(mapX[destIdx]);
//   int srcY = static_cast<int>(mapY[destIdx]);

//   if (srcX < srcW && srcY < srcH) {
//     int srcIdx = (srcY * srcW + srcX) * 3;
//     dest[destIdx * 3 + 0] = static_cast<T_out>(src[srcIdx + 0]);
//     dest[destIdx * 3 + 1] = static_cast<T_out>(src[srcIdx + 1]);
//     dest[destIdx * 3 + 2] = static_cast<T_out>(src[srcIdx + 2]);
//   } else {
//     dest[destIdx * 3 + 0] = defR;
//     dest[destIdx * 3 + 1] = defG;
//     dest[destIdx * 3 + 2] = defB;
//   }
// }

//------------------------------------------------------------------------------
// Templated Batched Remap Kernel for RGB Images (unchanged)
//------------------------------------------------------------------------------
// template <typename T_in, typename T_out>
// __global__ void BatchedRemapKernel(
//     const T_in* src,
//     int srcW,
//     int srcH,
//     T_out* dest,
//     int destW,
//     int destH,
//     const unsigned short* mapX,
//     const unsigned short* mapY,
//     T_out defR,
//     T_out defG,
//     T_out defB,
//     int batchSize) {
//   int b = blockIdx.z;
//   if (b >= batchSize)
//     return;

//   int srcImageSize = srcW * srcH * 3;
//   int destImageSize = destW * destH * 3;
//   int mapSize = destW * destH; // mapping arrays match destination size

//   const T_in* srcImage = src + b * srcImageSize;
//   T_out* destImage = dest + b * destImageSize;
//   const unsigned short* mapXImage = mapX + b * mapSize;
//   const unsigned short* mapYImage = mapY + b * mapSize;

//   int x = blockIdx.x * blockDim.x + threadIdx.x;
//   int y = blockIdx.y * blockDim.y + threadIdx.y;
//   if (x >= destW || y >= destH)
//     return;

//   int destIdx = y * destW + x;
//   int srcX = static_cast<int>(mapXImage[destIdx]);
//   int srcY = static_cast<int>(mapYImage[destIdx]);

//   if (srcX < srcW && srcY < srcH) {
//     int srcIdx = (srcY * srcW + srcX) * 3;
//     destImage[destIdx * 3 + 0] = static_cast<T_out>(srcImage[srcIdx + 0]);
//     destImage[destIdx * 3 + 1] = static_cast<T_out>(srcImage[srcIdx + 1]);
//     destImage[destIdx * 3 + 2] = static_cast<T_out>(srcImage[srcIdx + 2]);
//   } else {
//     destImage[destIdx * 3 + 0] = defR;
//     destImage[destIdx * 3 + 1] = defG;
//     destImage[destIdx * 3 + 2] = defB;
//   }
// }

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

  int srcImageSize = srcW * srcH;
  int destImageSize = destW * destH;

  const T_in* srcImage = src + b * srcImageSize;
  T_out* destImage = dest + b * destImageSize;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= destW || y >= destH)
    return;

  const int destIdx = y * destW + x;
  const int srcX = static_cast<int>(mapX[destIdx]);
  const int srcY = static_cast<int>(mapY[destIdx]);

  if (srcX < srcW && srcY < srcH) {
    int srcIdx = srcY * srcW + srcX;
    destImage[destIdx] = static_cast<T_out>(srcImage[srcIdx]);
  } else {
    destImage[destIdx] = deflt;
  }
}

//------------------------------------------------------------------------------
// NEW: Templated Batched Remap Kernel for RGB Images with Offset
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelOffset(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
    const unsigned short* mapX, // mapping arrays of size (remapW x remapH)
    const unsigned short* mapY,
    T_out defR,
    T_out defG,
    T_out defB,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int srcImageSize = srcW * srcH * 3;
  int destImageSize = destW * destH * 3;

  const T_in* srcImage = src + b * srcImageSize;
  T_out* destImage = dest + b * destImageSize;

  // Coordinates within the remap (sub-)region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  // Compute destination coordinates by adding offset.
  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= destW || destY < 0 || destY >= destH)
    return;

  int destIdx = destY * destW + destX;
  int mapIdx = y * remapW + x; // mapping arrays are sized remapW x remapH

  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if (srcX < srcW && srcY < srcH) {
    int srcIdx = (srcY * srcW + srcX) * 3;
    destImage[destIdx * 3 + 0] = static_cast<T_out>(srcImage[srcIdx + 0]);
    destImage[destIdx * 3 + 1] = static_cast<T_out>(srcImage[srcIdx + 1]);
    destImage[destIdx * 3 + 2] = static_cast<T_out>(srcImage[srcIdx + 2]);
  } else {
    destImage[destIdx * 3 + 0] = defR;
    destImage[destIdx * 3 + 1] = defG;
    destImage[destIdx * 3 + 2] = defB;
  }
}

template <typename T>
__device__ T* advance_bytes(T* current, ptrdiff_t diff) {
  return reinterpret_cast<T*>(reinterpret_cast<unsigned char*>(current) + diff);
}

template <typename T>
__device__ const T* advance_bytes(const T* current, ptrdiff_t diff) {
  return reinterpret_cast<const T*>(reinterpret_cast<const unsigned char*>(current) + diff);
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

  // int srcImageSize = srcW * srcH;
  // int destImageSize = destW * destH;

  // TODO: lots of these calculations dont need ot be done every time
  int destImageSizeBytes = dest.height * dest.pitch;

  // const T_in* srcImage = src + b * srcImageSize;
  // T_out* destImage = dest + b * destImageSize;

  T_out* destImage = advance_bytes(dest.d_ptr, b * destImageSizeBytes);

  // int destIdx = destY * destW + destX;
  T_out* dest_pos = advance_bytes(destImage, destY * dest.pitch) + destX;

  int mapIdx = y * remapW + x;

  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if (srcX < src.width && srcY < src.height) {
    // int srcIdx = srcY * srcW + srcX;
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    // destImage[destIdx] = cast_to<T_in, T_out>(srcImage[srcIdx]);
    *dest_pos = cast_to<T_in, T_out>(*src_pos);
  } else {
    // destImage[destIdx] = cast_to<T_in, T_out>(deflt);
    *dest_pos = cast_to<T_in, T_out>(deflt);
  }
}

template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetAdjust(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
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

  int srcImageSize = srcW * srcH;
  int destImageSize = destW * destH;

  const T_in* srcImage = src + b * srcImageSize;
  T_out* destImage = dest + b * destImageSize;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= destW || destY < 0 || destY >= destH)
    return;

  int destIdx = destY * destW + destX;
  int mapIdx = y * remapW + x;

  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if (srcX < srcW && srcY < srcH) {
    int srcIdx = srcY * srcW + srcX;
    destImage[destIdx] = PixelAdjuster<T_out>::adjust(static_cast<T_out>(srcImage[srcIdx]), adjustment);
  } else {
    // We trust that srcY will also be kUnmappedPositionValue
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      destImage[destIdx] = deflt;
    }
  }
}

//------------------------------------------------------------------------------
// NEW: Templated Batched Remap Kernel EX with Offset (Single-channel)
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetWithDestMap(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
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

  int srcImageSize = srcW * srcH;
  int destImageSize = destW * destH;

  const T_in* srcImage = src + b * srcImageSize;
  T_out* destImage = dest + b * destImageSize;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= destW || destY < 0 || destY >= destH)
    return;

  int destIdx = destY * destW + destX;
  int mapIdx = y * remapW + x;

  int checkIdx = (offsetY + y) * destW + (offsetX + x);

  if (dest_image_map[checkIdx] == this_image_index) {
    int srcX = static_cast<int>(mapX[mapIdx]);
    int srcY = static_cast<int>(mapY[mapIdx]);

    if (srcX < srcW && srcY < srcH) {
      int srcIdx = srcY * srcW + srcX;
      destImage[destIdx] = static_cast<T_out>(srcImage[srcIdx]);
    } else {
      destImage[destIdx] = deflt;
    }
  }
}

template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetWithDestMapAdjust(
    const T_in* src,
    int srcW,
    int srcH,
    T_out* dest,
    int destW,
    int destH,
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

  int srcImageSize = srcW * srcH;
  int destImageSize = destW * destH;

  const T_in* srcImage = src + b * srcImageSize;
  T_out* destImage = dest + b * destImageSize;

  // Coordinates within the remap region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= remapW || y >= remapH)
    return;

  int destX = offsetX + x;
  int destY = offsetY + y;
  if (destX < 0 || destX >= destW || destY < 0 || destY >= destH)
    return;

  int destIdx = destY * destW + destX;
  int mapIdx = y * remapW + x;

  int checkIdx = (offsetY + y) * destW + (offsetX + x);

  if (dest_image_map[checkIdx] == this_image_index) {
    int srcX = static_cast<int>(mapX[mapIdx]);
    int srcY = static_cast<int>(mapY[mapIdx]);
    if (srcX < srcW && srcY < srcH) {
      int srcIdx = srcY * srcW + srcX;
      // Out is more likely to be a float, so adjust after any cast
      destImage[destIdx] = PixelAdjuster<T_out>::adjust(static_cast<T_out>(srcImage[srcIdx]), adjustment);
    } else {
      destImage[destIdx] = deflt;
    }
  }
}

} // anonymous namespace

//------------------------------------------------------------------------------
// Host Function: Remap a Single Image (unchanged)
//------------------------------------------------------------------------------
// template <typename T_in, typename T_out>
// cudaError_t remap_kernel(
//     const T_in* d_src,
//     int srcW,
//     int srcH,
//     T_out* d_dest,
//     int destW,
//     int destH,
//     const unsigned short* d_mapX,
//     const unsigned short* d_mapY,
//     T_out defR,
//     T_out defG,
//     T_out defB,
//     cudaStream_t stream) {
//   dim3 blockDim(16, 16);
//   dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y);
//   remapKernel<T_in, T_out>
//       <<<gridDim, blockDim, 0, stream>>>(d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB);
//   return cudaGetLastError();
// }

//------------------------------------------------------------------------------
// Host Function: Batched Remap (unchanged)
//------------------------------------------------------------------------------
// template <typename T_in, typename T_out>
// cudaError_t batched_remap_kernel(
//     const T_in* d_src,
//     int srcW,
//     int srcH,
//     T_out* d_dest,
//     int destW,
//     int destH,
//     const unsigned short* d_mapX,
//     const unsigned short* d_mapY,
//     T_in defR,
//     T_in defG,
//     T_in defB,
//     int batchSize,
//     cudaStream_t stream) {
//   dim3 blockDim(16, 16, 1);
//   dim3 gridDim((destW + blockDim.x - 1) / blockDim.x, (destH + blockDim.y - 1) / blockDim.y, batchSize);
//   BatchedRemapKernel<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
//       d_src, srcW, srcH, d_dest, destW, destH, d_mapX, d_mapY, defR, defG, defB, batchSize);
//   return cudaGetLastError();
// }

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
// NEW: Host Function: Batched Remap with Offset (RGB)
//------------------------------------------------------------------------------

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_offset(
    const T_in* d_src,
    int srcW,
    int srcH,
    // int srcPitch,
    T_out* d_dest,
    int destW,
    int destH,
    // int destPitch,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in defR,
    T_in defG,
    T_in defB,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelOffset<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      d_src,
      srcW,
      srcH,
      d_dest,
      destW,
      destH,
      d_mapX,
      d_mapY,
      defR,
      defG,
      defB,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// Host Function: Batched Remap EX with Offset (Single-channel)
//------------------------------------------------------------------------------

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset(
    const CudaSurface<T_in> src,
    CudaSurface<T_out> dest,
    const unsigned short* d_mapX,
    const unsigned short* d_mapY,
    T_in deflt,
    int batchSize,
    int remapW,
    int remapH,
    int offsetX,
    int offsetY,
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((remapW + blockDim.x - 1) / blockDim.x, (remapH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffset<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
      src,
      dest,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY);
  return cudaGetLastError();
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_adjust(
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
      d_src,
      srcW,
      srcH,
      d_dest,
      destW,
      destH,
      d_mapX,
      d_mapY,
      deflt,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY,
      no_unmapped_write,
      adjustment);
  return cudaGetLastError();
}

//------------------------------------------------------------------------------
// NEW: Host Function: Batched Remap EX with Offset (Single-channel)
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
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
      d_src,
      srcW,
      srcH,
      d_dest,
      destW,
      destH,
      d_mapX,
      d_mapY,
      deflt,
      this_image_index,
      dest_image_map,
      batchSize,
      remapW,
      remapH,
      offsetX,
      offsetY);
  return cudaGetLastError();
}

template <typename T_in, typename T_out>
cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust(
    const T_in* d_src,
    int srcW,
    int srcH,
    T_out* d_dest,
    int destW,
    int destH,
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
      d_src,
      srcW,
      srcH,
      d_dest,
      destW,
      destH,
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

// Macro for instantiating batched_remap_kernel_offset<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_offset<Tin, Tout>( \
      const Tin* d_src,                                        \
      int srcW,                                                \
      int srcH,                                                \
      Tout* d_dest,                                            \
      int destW,                                               \
      int destH,                                               \
      const unsigned short* d_mapX,                            \
      const unsigned short* d_mapY,                            \
      Tin defR,                                                \
      Tin defG,                                                \
      Tin defB,                                                \
      int batchSize,                                           \
      int remapW,                                              \
      int remapH,                                              \
      int offsetX,                                             \
      int offsetY,                                             \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex_offset<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex_offset<Tin, Tout>( \
      const CudaSurface<Tin> src,                                 \
      CudaSurface<Tout> dest,                                     \
      const unsigned short* d_mapX,                               \
      const unsigned short* d_mapY,                               \
      Tin deflt,                                                  \
      int batchSize,                                              \
      int remapW,                                                 \
      int remapH,                                                 \
      int offsetX,                                                \
      int offsetY,                                                \
      cudaStream_t stream);

#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex_offset_adjust<Tin, Tout>( \
      const Tin* d_src,                                                  \
      int srcW,                                                          \
      int srcH,                                                          \
      Tout* d_dest,                                                      \
      int destW,                                                         \
      int destH,                                                         \
      const unsigned short* d_mapX,                                      \
      const unsigned short* d_mapY,                                      \
      Tin deflt,                                                         \
      int batchSize,                                                     \
      int remapW,                                                        \
      int remapH,                                                        \
      int offsetX,                                                       \
      int offsetY,                                                       \
      bool no_unmapped_write,                                            \
      float3 adjustment,                                                 \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL(Tin, Tout)     \
  template cudaError_t batched_remap_kernel<Tin, Tout>( \
      const Tin* d_src,                                 \
      int srcW,                                         \
      int srcH,                                         \
      Tout* d_dest,                                     \
      int destW,                                        \
      int destH,                                        \
      const unsigned short* d_mapX,                     \
      const unsigned short* d_mapY,                     \
      Tin defR,                                         \
      Tin defG,                                         \
      Tin defB,                                         \
      int batchSize,                                    \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex<Tin, Tout>( \
      const Tin* d_src,                                    \
      int srcW,                                            \
      int srcH,                                            \
      Tout* d_dest,                                        \
      int destW,                                           \
      int destH,                                           \
      const unsigned short* d_mapX,                        \
      const unsigned short* d_mapY,                        \
      Tin deflt,                                           \
      int batchSize,                                       \
      cudaStream_t stream);

#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_ADJUST(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex_adjust<Tin, Tout>( \
      const Tin* d_src,                                           \
      int srcW,                                                   \
      int srcH,                                                   \
      Tout* d_dest,                                               \
      int destW,                                                  \
      int destH,                                                  \
      const unsigned short* d_mapX,                               \
      const unsigned short* d_mapY,                               \
      Tin deflt,                                                  \
      int batchSize,                                              \
      float3 adjustment cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex_offset_with_dest_map<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex_offset_with_dest_map<Tin, Tout>( \
      const Tin* d_src,                                                         \
      int srcW,                                                                 \
      int srcH,                                                                 \
      Tout* d_dest,                                                             \
      int destW,                                                                \
      int destH,                                                                \
      const unsigned short* d_mapX,                                             \
      const unsigned short* d_mapY,                                             \
      Tin deflt,                                                                \
      int this_image_index,                                                     \
      const unsigned char* dest_image_map,                                      \
      int batchSize,                                                            \
      int remapW,                                                               \
      int remapH,                                                               \
      int offsetX,                                                              \
      int offsetY,                                                              \
      cudaStream_t stream);

// Macro for instantiating batched_remap_kernel_ex_offset_with_dest_map_adjust<T_in, T_out>
#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP_ADJUST(Tin, Tout)     \
  template cudaError_t batched_remap_kernel_ex_offset_with_dest_map_adjust<Tin, Tout>( \
      const Tin* d_src,                                                                \
      int srcW,                                                                        \
      int srcH,                                                                        \
      Tout* d_dest,                                                                    \
      int destW,                                                                       \
      int destH,                                                                       \
      const unsigned short* d_mapX,                                                    \
      const unsigned short* d_mapY,                                                    \
      Tin deflt,                                                                       \
      int this_image_index,                                                            \
      const unsigned char* dest_image_map,                                             \
      int batchSize,                                                                   \
      int remapW,                                                                      \
      int remapH,                                                                      \
      int offsetX,                                                                     \
      int offsetY,                                                                     \
      float3 adjustment,                                                               \
      cudaStream_t stream);

// For batched_remap_kernel_offset
INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(float, float)
INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(uchar1, uchar1)
INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(float, __half)
INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(__half, float)
INSTANTIATE_BATCHED_REMAP_KERNEL_OFFSET(__half, __half)

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

// For batched_remap_kernel
// INSTANTIATE_BATCHED_REMAP_KERNEL(float, float)
// INSTANTIATE_BATCHED_REMAP_KERNEL(float, __half)
// INSTANTIATE_BATCHED_REMAP_KERNEL(__half, float)
// INSTANTIATE_BATCHED_REMAP_KERNEL(__half, __half)
// INSTANTIATE_BATCHED_REMAP_KERNEL(uchar1, uchar1)

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
