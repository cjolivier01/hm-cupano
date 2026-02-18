#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "cudaImageAdjust.cuh"
#include "cudaRemap.h" // Assumed to declare these host functions

#include "cudaUtils.cuh"

#include <cassert>
#include <limits>

using namespace hm::cupano::cuda;
namespace {

constexpr unsigned short kUnmappedPositionValue = std::numeric_limits<unsigned short>::max();

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

  if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
    if (no_unmapped_write && srcX == kUnmappedPositionValue) {
      // Don't write anything for alpha 0
      return;
    }
  }

  if (srcX < src.width && srcY < src.height) {
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    *dest_pos = perform_cast<T_out>(*src_pos);
  } else {
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      *dest_pos = perform_cast<T_out>(deflt);
      if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
        // Has an alpha channel, so clear it
        if (srcX == kUnmappedPositionValue) {
          dest_pos->w = 0;
        } else {
          dest_pos->w = BaseScalar_t<T_out>(255);
        }
      }
    }
  }
}

//------------------------------------------------------------------------------
// NEW: Templated Batched Remap Kernel EX with Offset over a ROI in remap coords
//------------------------------------------------------------------------------
template <typename T_in, typename T_out>
__global__ void BatchedRemapKernelExOffsetROI(
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
    int roiX,
    int roiY,
    int roiW,
    int roiH,
    bool no_unmapped_write) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  // Coordinates within the ROI region.
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= roiW || y >= roiH)
    return;

  const int mapXpos = roiX + x;
  const int mapYpos = roiY + y;
  if (mapXpos < 0 || mapXpos >= remapW || mapYpos < 0 || mapYpos >= remapH)
    return;

  const int destX = offsetX + mapXpos;
  const int destY = offsetY + mapYpos;
  if (destX < 0 || destX >= dest.width || destY < 0 || destY >= dest.height)
    return;

  int destImageSizeBytes = dest.height * dest.pitch;

  T_out* destImage = advance_bytes(dest.d_ptr, b * destImageSizeBytes);
  T_out* dest_pos = advance_bytes(destImage, destY * dest.pitch) + destX;

  int mapIdx = mapYpos * remapW + mapXpos;
  int srcX = static_cast<int>(mapX[mapIdx]);
  int srcY = static_cast<int>(mapY[mapIdx]);

  if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
    if (no_unmapped_write && srcX == kUnmappedPositionValue) {
      // Don't write anything for alpha 0.
      return;
    }
  }

  if (srcX < src.width && srcY < src.height) {
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    *dest_pos = perform_cast<T_out>(*src_pos);
  } else {
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      *dest_pos = perform_cast<T_out>(deflt);
      if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
        if (srcX == kUnmappedPositionValue) {
          dest_pos->w = 0;
        } else {
          dest_pos->w = BaseScalar_t<T_out>(255);
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

  // if constexpr (sizeof(T_out) / sizeof(BaseScalar_t<T_out>) == 4) {
  //   if (no_unmapped_write && srcX == kUnmappedPositionValue) {
  //     // Don't write anything for alpha 0
  //     return;
  //   }
  // }

  if (srcX < src.width && srcY < src.height) {
    int srcImageSizeBytes = src.height * src.pitch;
    const T_in* srcImage = advance_bytes(src.d_ptr, b * srcImageSizeBytes);
    const T_in* src_pos = advance_bytes(srcImage, srcY * src.pitch) + srcX;
    *dest_pos = PixelAdjuster<T_out>::adjust(static_cast<T_out>(*src_pos), adjustment);
  } else {
    if (!no_unmapped_write || srcX != kUnmappedPositionValue) {
      *dest_pos = perform_cast<T_out>(deflt);
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
  int checkIdx = destY * dest.width + destX;
  if (dest_image_map[checkIdx] == this_image_index) {
    int mapIdx = y * remapW + x;
    int srcX = static_cast<int>(mapX[mapIdx]);
    int srcY = static_cast<int>(mapY[mapIdx]);

    if (srcX < src.width && srcY < src.height) {
      const T_in* src_pos = surface_ptr(src, b, srcX, srcY);
      *surface_ptr(dest, b, destX, destY) = static_cast<T_out>(*src_pos);
    } else {
      *surface_ptr(dest, b, destX, destY) = deflt;
    }
  }
}

//------------------------------------------------------------------------------
// NEW: Fused hard-seam remap for N images (select per-pixel input by dest map)
//------------------------------------------------------------------------------
template <typename T>
__global__ void BatchedRemapHardSeamKernelN(
    const CudaSurface<T>* inputs, // length n_images
    const unsigned short* const* mapX_ptrs, // length n_images
    const unsigned short* const* mapY_ptrs, // length n_images
    const int2* offsets, // length n_images
    const int2* sizes, // length n_images (w,h)
    int n_images,
    const unsigned char* dest_image_map, // [H×W] indexed [0..n_images-1]
    CudaSurface<T> dest,
    int batchSize) {
  int b = blockIdx.z;
  if (b >= batchSize)
    return;

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= static_cast<int>(dest.width) || y >= static_cast<int>(dest.height))
    return;

  const int maskIdx = y * static_cast<int>(dest.width) + x;
  const int imageIndex = static_cast<int>(dest_image_map[maskIdx]);
  if (imageIndex < 0 || imageIndex >= n_images)
    return;

  const int2 off = offsets[imageIndex];
  const int2 sz = sizes[imageIndex];
  const int rx = x - off.x;
  const int ry = y - off.y;
  if (rx < 0 || ry < 0 || rx >= sz.x || ry >= sz.y)
    return;

  const int mapIdx = ry * sz.x + rx;
  const int srcX = static_cast<int>(mapX_ptrs[imageIndex][mapIdx]);
  const int srcY = static_cast<int>(mapY_ptrs[imageIndex][mapIdx]);

  if (srcX == kUnmappedPositionValue)
    return;

  const CudaSurface<T> src = inputs[imageIndex];
  if (srcX < static_cast<int>(src.width) && srcY < static_cast<int>(src.height)) {
    *surface_ptr(dest, b, x, y) = *surface_ptr(src, b, srcX, srcY);
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
    cudaStream_t stream) {
  if (roiW <= 0 || roiH <= 0)
    return cudaSuccess;
  if (roiX < 0 || roiY < 0 || roiW < 0 || roiH < 0)
    return cudaErrorInvalidValue;
  if (roiX + roiW > remapW || roiY + roiH > remapH)
    return cudaErrorInvalidValue;

  dim3 blockDim(16, 16, 1);
  dim3 gridDim((roiW + blockDim.x - 1) / blockDim.x, (roiH + blockDim.y - 1) / blockDim.y, batchSize);
  BatchedRemapKernelExOffsetROI<T_in, T_out><<<gridDim, blockDim, 0, stream>>>(
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
      no_unmapped_write);
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
    cudaStream_t stream) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim(
      (static_cast<int>(dest.width) + blockDim.x - 1) / blockDim.x,
      (static_cast<int>(dest.height) + blockDim.y - 1) / blockDim.y,
      batchSize);
  BatchedRemapHardSeamKernelN<T><<<gridDim, blockDim, 0, stream>>>(
      d_inputs, d_mapX_ptrs, d_mapY_ptrs, d_offsets, d_sizes, n_images, dest_image_map, dest, batchSize);
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

#define INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(T_in, T_out)     \
  template cudaError_t batched_remap_kernel_ex_offset_roi<T_in, T_out>( \
      const CudaSurface<T_in>& src,                                     \
      const CudaSurface<T_out>& dest,                                   \
      const unsigned short* d_mapX,                                     \
      const unsigned short* d_mapY,                                     \
      T_in deflt,                                                       \
      int batchSize,                                                    \
      int remapW,                                                       \
      int remapH,                                                       \
      int offsetX,                                                      \
      int offsetY,                                                      \
      int roiX,                                                         \
      int roiY,                                                         \
      int roiW,                                                         \
      int roiH,                                                         \
      bool no_unmapped_write,                                           \
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
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(float1, float1)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(uchar4, uchar4)
// INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET(__half, __half)

INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(uchar3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(float1, float1)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(uchar4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ROI(uchar4, uchar4)

INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(float3, float3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(uchar3, uchar3)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(float4, float4)
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(uchar4, uchar4)
// INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_ADJUST(uchar4, float4)

// For batched_remap_kernel_ex
INSTANTIATE_BATCHED_REMAP_KERNEL_EX(float3, float3)

// Instantiate for float input and float output:
INSTANTIATE_BATCHED_REMAP_KERNEL_EX_OFFSET_WITH_DEST_MAP(float1, float1)
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

#define INSTANTIATE_BATCHED_REMAP_HARD_SEAM_N(T)         \
  template cudaError_t batched_remap_hard_seam_kernel_n( \
      const CudaSurface<T>*,                             \
      const unsigned short* const*,                      \
      const unsigned short* const*,                      \
      const int2*,                                       \
      const int2*,                                       \
      int,                                               \
      const unsigned char*,                              \
      CudaSurface<T>,                                    \
      int,                                               \
      cudaStream_t);

INSTANTIATE_BATCHED_REMAP_HARD_SEAM_N(uchar3)
INSTANTIATE_BATCHED_REMAP_HARD_SEAM_N(uchar4)
INSTANTIATE_BATCHED_REMAP_HARD_SEAM_N(float3)
INSTANTIATE_BATCHED_REMAP_HARD_SEAM_N(float4)
