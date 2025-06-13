#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include "cudaFusedKernels3.h"
#include "cudaImageAdjust.cuh"
#include "cudaUtils.cuh"
#include "cupano/pano/canvasManager3.h"

namespace hm {
namespace pano {
namespace cuda {

using hm::cupano::cuda::perform_cast;

namespace {

constexpr unsigned short kUnmappedPositionValue = std::numeric_limits<unsigned short>::max();

/**
 * Fused kernel that remaps all three images directly to their full buffers
 * Eliminates intermediate canvas operations for soft seam mode
 */
template <typename T_pipeline, typename T_compute>
__global__ void FusedRemapToFullKernel3(
    // Input surfaces
    CudaSurface<T_pipeline> inputImage0,
    CudaSurface<T_pipeline> inputImage1,
    CudaSurface<T_pipeline> inputImage2,
    // Remap coordinates
    const uint16_t* remap_0_x,
    const uint16_t* remap_0_y,
    const uint16_t* remap_1_x,
    const uint16_t* remap_1_y,
    const uint16_t* remap_2_x,
    const uint16_t* remap_2_y,
    // Remap dimensions
    int remap0_w,
    int remap0_h,
    int remap1_w,
    int remap1_h,
    int remap2_w,
    int remap2_h,
    // Output full buffers
    CudaSurface<T_compute> cudaFull0,
    CudaSurface<T_compute> cudaFull1,
    CudaSurface<T_compute> cudaFull2,
    // ROI parameters
    // int roi_width,
    // int roi_height,
    int offset0_x,
    int offset0_y,
    int offset1_x,
    int offset1_y,
    int offset2_x,
    int offset2_y,
    // Canvas positions
    int canvas0_x,
    int canvas0_y,
    int canvas1_x,
    int canvas1_y,
    int canvas2_x,
    int canvas2_y,
    // Adjustments
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment) {
  int b = blockIdx.z;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 0 || y < 0 || x >= cudaFull0.width || y >= cudaFull0.height)
    return;

  // Process image 0
  {
    int full_x = x + offset0_x;
    int full_y = y + offset0_y;

    if (full_x >= 0 && full_x < cudaFull0.width && full_y >= 0 && full_y < cudaFull0.height) {
      int remap_x = full_x - canvas0_x;
      int remap_y = full_y - canvas0_y;

      if (remap_x >= 0 && remap_x < remap0_w && remap_y >= 0 && remap_y < remap0_h) {
        int mapIdx = remap_y * remap0_w + remap_x;
        int srcX = remap_0_x[mapIdx];
        int srcY = remap_0_y[mapIdx];

        if (srcX != kUnmappedPositionValue && srcX < inputImage0.width && srcY < inputImage0.height) {
          T_pipeline* src_ptr = surface_ptr(inputImage0, b, srcX, srcY);
          T_compute out_pixel = perform_cast<T_compute>(*src_ptr);

          if (apply_adjustment) {
            out_pixel = PixelAdjuster<T_compute>::adjust(out_pixel, adjustment0);
          }

          *surface_ptr(cudaFull0, b, x, y) = out_pixel;
        }
      }
    }
  }

  // Process image 1
  if (x >= 0 && x >= canvas1_x && x < canvas1_x + remap1_w && y >= canvas1_y && y < canvas1_y + remap1_h) {
    // int full_x = x + offset1_x;
    // int full_y = y + offset1_y;
    // if (full_x >= 0 && full_x < cudaFull1.width && full_y >= 0 && full_y < cudaFull1.height) {
    //  int remap_x = full_x - canvas1_x;
    //  int remap_y = full_y - canvas1_y;
    int remap_x = x - canvas1_x;
    int remap_y = y - canvas1_y;

    if (remap_x >= 0 && remap_x < remap1_w && remap_y >= 0 && remap_y < remap1_h) {
      int mapIdx = remap_y * remap1_w + remap_x;
      int srcX = remap_1_x[mapIdx];
      int srcY = remap_1_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage1.width && srcY < inputImage1.height) {
        T_pipeline* src_ptr = surface_ptr(inputImage1, b, srcX, srcY);
        T_compute out_pixel = perform_cast<T_compute>(*src_ptr);

        if (apply_adjustment) {
          out_pixel = PixelAdjuster<T_compute>::adjust(out_pixel, adjustment1);
        }
        // printf("image 1 dest x: %d/%d\n", (int)x, (int)cudaFull1.width);
        *surface_ptr(cudaFull1, b, x, y) = out_pixel;
      }
    }
    //}
  }

  // Process image 2
  {
    int full_x = x + offset2_x;
    int full_y = y + offset2_y;

    if (full_x >= 0 && full_x < cudaFull2.width && full_y >= 0 && full_y < cudaFull2.height) {
      int remap_x = full_x - canvas2_x;
      int remap_y = full_y - canvas2_y;

      if (remap_x >= 0 && remap_x < remap2_w && remap_y >= 0 && remap_y < remap2_h) {
        int mapIdx = remap_y * remap2_w + remap_x;
        int srcX = remap_2_x[mapIdx];
        int srcY = remap_2_y[mapIdx];

        if (srcX != kUnmappedPositionValue && srcX < inputImage2.width && srcY < inputImage2.height) {
          T_pipeline* src_ptr = surface_ptr(inputImage2, b, srcX, srcY);
          T_compute out_pixel = perform_cast<T_compute>(*src_ptr);

          if (apply_adjustment) {
            out_pixel = PixelAdjuster<T_compute>::adjust(out_pixel, adjustment2);
          }

          *surface_ptr(cudaFull2, b, x, y) = out_pixel;
        }
      }
    }
  }
}

/**
 * Fused kernel that remaps all three images to canvas for non-blended regions
 */
template <typename T_pipeline>
__global__ void FusedRemapToCanvasKernel3(
    // Input surfaces
    CudaSurface<T_pipeline> inputImage0,
    CudaSurface<T_pipeline> inputImage1,
    CudaSurface<T_pipeline> inputImage2,
    // Remap coordinates
    const uint16_t* remap_0_x,
    const uint16_t* remap_0_y,
    const uint16_t* remap_1_x,
    const uint16_t* remap_1_y,
    const uint16_t* remap_2_x,
    const uint16_t* remap_2_y,
    // Remap dimensions
    int remap0_w,
    int remap0_h,
    int remap1_w,
    int remap1_h,
    int remap2_w,
    int remap2_h,
    // Output canvas
    CudaSurface<T_pipeline> canvas,
    // Canvas positions
    int canvas0_x,
    int canvas0_y,
    int canvas1_x,
    int canvas1_y,
    int canvas2_x,
    int canvas2_y,
    // Blend region to avoid (soft seam area)
    int blend_x_start,
    int blend_y_start,
    int blend_width,
    int blend_height,
    // Adjustments
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment) {
  int b = blockIdx.z;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= canvas.width || y >= canvas.height)
    return;

  // Skip blend region - it will be handled by Laplacian blending
  if (x >= blend_x_start && x < blend_x_start + blend_width && y >= blend_y_start && y < blend_y_start + blend_height) {
    return;
  }

  bool pixel_set = false;
  T_pipeline pixel_value;

  // Check image 0
  if (!pixel_set) {
    int remap_x = x - canvas0_x;
    int remap_y = y - canvas0_y;

    if (remap_x >= 0 && remap_x < remap0_w && remap_y >= 0 && remap_y < remap0_h) {
      int mapIdx = remap_y * remap0_w + remap_x;
      int srcX = remap_0_x[mapIdx];
      int srcY = remap_0_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage0.width && srcY < inputImage0.height) {
        pixel_value = *surface_ptr(inputImage0, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment0);
        }
        pixel_set = true;
      }
    }
  }

  // Check image 1
  if (!pixel_set) {
    int remap_x = x - canvas1_x;
    int remap_y = y - canvas1_y;

    if (remap_x >= 0 && remap_x < remap1_w && remap_y >= 0 && remap_y < remap1_h) {
      int mapIdx = remap_y * remap1_w + remap_x;
      int srcX = remap_1_x[mapIdx];
      int srcY = remap_1_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage1.width && srcY < inputImage1.height) {
        pixel_value = *surface_ptr(inputImage1, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment1);
        }
        pixel_set = true;
      }
    }
  }

  // Check image 2
  if (!pixel_set) {
    int remap_x = x - canvas2_x;
    int remap_y = y - canvas2_y;

    if (remap_x >= 0 && remap_x < remap2_w && remap_y >= 0 && remap_y < remap2_h) {
      int mapIdx = remap_y * remap2_w + remap_x;
      int srcX = remap_2_x[mapIdx];
      int srcY = remap_2_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage2.width && srcY < inputImage2.height) {
        pixel_value = *surface_ptr(inputImage2, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment2);
        }
        pixel_set = true;
      }
    }
  }

  if (pixel_set) {
    *surface_ptr(canvas, b, x, y) = pixel_value;
  }
}

/**
 * Fused kernel for hard seam - processes all three images based on mask
 */
template <typename T_pipeline>
__global__ void FusedRemapHardSeam3(
    // Input surfaces
    CudaSurface<T_pipeline> inputImage0,
    CudaSurface<T_pipeline> inputImage1,
    CudaSurface<T_pipeline> inputImage2,
    // Remap coordinates
    const uint16_t* remap_0_x,
    const uint16_t* remap_0_y,
    const uint16_t* remap_1_x,
    const uint16_t* remap_1_y,
    const uint16_t* remap_2_x,
    const uint16_t* remap_2_y,
    // Remap dimensions
    int remap0_w,
    int remap0_h,
    int remap1_w,
    int remap1_h,
    int remap2_w,
    int remap2_h,
    // Hard seam mask
    const unsigned char* hardSeamMask,
    // Output canvas
    CudaSurface<T_pipeline> canvas,
    // Canvas positions
    int canvas0_x,
    int canvas0_y,
    int canvas1_x,
    int canvas1_y,
    int canvas2_x,
    int canvas2_y,
    // Adjustments
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment) {
  int b = blockIdx.z;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= canvas.width || y >= canvas.height)
    return;

  // Check which image should contribute to this pixel
  int maskIdx = y * canvas.width + x;
  unsigned char imageIndex = hardSeamMask[maskIdx];

  T_pipeline pixel_value;
  bool pixel_set = false;

  // Process based on which image owns this pixel
  if (imageIndex == 0) {
    int remap_x = x - canvas0_x;
    int remap_y = y - canvas0_y;

    if (remap_x >= 0 && remap_x < remap0_w && remap_y >= 0 && remap_y < remap0_h) {
      int mapIdx = remap_y * remap0_w + remap_x;
      int srcX = remap_0_x[mapIdx];
      int srcY = remap_0_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage0.width && srcY < inputImage0.height) {
        pixel_value = *surface_ptr(inputImage0, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment0);
        }
        pixel_set = true;
      }
    }
  } else if (imageIndex == 1) {
    int remap_x = x - canvas1_x;
    int remap_y = y - canvas1_y;

    if (remap_x >= 0 && remap_x < remap1_w && remap_y >= 0 && remap_y < remap1_h) {
      int mapIdx = remap_y * remap1_w + remap_x;
      int srcX = remap_1_x[mapIdx];
      int srcY = remap_1_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage1.width && srcY < inputImage1.height) {
        pixel_value = *surface_ptr(inputImage1, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment1);
        }
        pixel_set = true;
      }
    }
  } else if (imageIndex == 2) {
    int remap_x = x - canvas2_x;
    int remap_y = y - canvas2_y;

    if (remap_x >= 0 && remap_x < remap2_w && remap_y >= 0 && remap_y < remap2_h) {
      int mapIdx = remap_y * remap2_w + remap_x;
      int srcX = remap_2_x[mapIdx];
      int srcY = remap_2_y[mapIdx];

      if (srcX != kUnmappedPositionValue && srcX < inputImage2.width && srcY < inputImage2.height) {
        pixel_value = *surface_ptr(inputImage2, b, srcX, srcY);
        if (apply_adjustment) {
          pixel_value = PixelAdjuster<T_pipeline>::adjust(pixel_value, adjustment2);
        }
        pixel_set = true;
      }
    }
  }

  if (pixel_set) {
    *surface_ptr(canvas, b, x, y) = pixel_value;
  }
}

} // anonymous namespace

// Host function implementations

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
  dim3 block(16, 16);
  dim3 grid(
      (cudaFull0.width() + block.x - 1) / block.x,
      (cudaFull0.height() + block.y - 1) / block.y,
      inputImage0.batch_size());

  const auto& canvas_positions = canvas_manager.canvas_positions();

  FusedRemapToFullKernel3<T_pipeline, T_compute><<<grid, block, 0, stream>>>(
      inputImage0.surface(),
      inputImage1.surface(),
      inputImage2.surface(),
      remap_0_x.data(),
      remap_0_y.data(),
      remap_1_x.data(),
      remap_1_y.data(),
      remap_2_x.data(),
      remap_2_y.data(),
      remap_0_x.width(),
      remap_0_x.height(),
      remap_1_x.width(),
      remap_1_x.height(),
      remap_2_x.width(),
      remap_2_x.height(),
      cudaFull0.surface(),
      cudaFull1.surface(),
      cudaFull2.surface(),
      // /*roi_width= GET RID OF*/cudaFull2.width(),
      // /*roi_height= GET RID OF*/cudaFull2.height(),
      // canvas_manager.remapped_image_roi_blend_0.width,
      // canvas_manager.remapped_image_roi_blend_0.height,
      canvas_manager._remapper_0.xpos,
      0,
      canvas_manager._remapper_1.xpos,
      0,
      canvas_manager._remapper_2.xpos,
      0,
      canvas_positions[0].x,
      canvas_positions[0].y,
      canvas_positions[1].x,
      canvas_positions[1].y,
      canvas_positions[2].x,
      canvas_positions[2].y,
      adjustment0,
      adjustment1,
      adjustment2,
      apply_adjustment);

  return CudaStatus(cudaGetLastError());
}

template <typename T_pipeline>
CudaStatus launchFusedRemapToCanvasKernel3(
    const CudaMat<T_pipeline>& inputImage0,
    const CudaMat<T_pipeline>& inputImage1,
    const CudaMat<T_pipeline>& inputImage2,
    const CudaMat<uint16_t>& remap_0_x,
    const CudaMat<uint16_t>& remap_0_y,
    const CudaMat<uint16_t>& remap_1_x,
    const CudaMat<uint16_t>& remap_1_y,
    const CudaMat<uint16_t>& remap_2_x,
    const CudaMat<uint16_t>& remap_2_y,
    CudaMat<T_pipeline>& canvas,
    const CanvasManager3& canvas_manager,
    float3 adjustment0,
    float3 adjustment1,
    float3 adjustment2,
    bool apply_adjustment,
    cudaStream_t stream) {
  dim3 block(16, 16);
  dim3 grid(
      (canvas.width() + block.x - 1) / block.x, (canvas.height() + block.y - 1) / block.y, inputImage0.batch_size());

  FusedRemapToCanvasKernel3<T_pipeline><<<grid, block, 0, stream>>>(
      inputImage0.surface(),
      inputImage1.surface(),
      inputImage2.surface(),
      remap_0_x.data(),
      remap_0_y.data(),
      remap_1_x.data(),
      remap_1_y.data(),
      remap_2_x.data(),
      remap_2_y.data(),
      remap_0_x.width(),
      remap_0_x.height(),
      remap_1_x.width(),
      remap_1_x.height(),
      remap_2_x.width(),
      remap_2_x.height(),
      canvas.surface(),
      canvas_manager.canvas_positions()[0].x,
      canvas_manager.canvas_positions()[0].y,
      canvas_manager.canvas_positions()[1].x,
      canvas_manager.canvas_positions()[1].y,
      canvas_manager.canvas_positions()[2].x,
      canvas_manager.canvas_positions()[2].y,
      canvas_manager._x_blend_start,
      canvas_manager._y_blend_start,
      canvas_manager.remapped_image_roi_blend_0.width,
      canvas_manager.remapped_image_roi_blend_0.height,
      adjustment0,
      adjustment1,
      adjustment2,
      apply_adjustment);

  return CudaStatus(cudaGetLastError());
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
  dim3 block(16, 16);
  dim3 grid(
      (canvas.width() + block.x - 1) / block.x, (canvas.height() + block.y - 1) / block.y, inputImage0.batch_size());

  FusedRemapHardSeam3<T_pipeline><<<grid, block, 0, stream>>>(
      inputImage0.surface(),
      inputImage1.surface(),
      inputImage2.surface(),
      remap_0_x.data(),
      remap_0_y.data(),
      remap_1_x.data(),
      remap_1_y.data(),
      remap_2_x.data(),
      remap_2_y.data(),
      remap_0_x.width(),
      remap_0_x.height(),
      remap_1_x.width(),
      remap_1_x.height(),
      remap_2_x.width(),
      remap_2_x.height(),
      hardSeamMask.data(),
      canvas.surface(),
      canvas_manager.canvas_positions()[0].x,
      canvas_manager.canvas_positions()[0].y,
      canvas_manager.canvas_positions()[1].x,
      canvas_manager.canvas_positions()[1].y,
      canvas_manager.canvas_positions()[2].x,
      canvas_manager.canvas_positions()[2].y,
      adjustment0,
      adjustment1,
      adjustment2,
      apply_adjustment);

  return CudaStatus(cudaGetLastError());
}

// Explicit template instantiations
#define INSTANTIATE_FUSED_KERNELS(T_pipeline, T_compute)                    \
  template CudaStatus launchFusedRemapToFullKernel3<T_pipeline, T_compute>( \
      const CudaMat<T_pipeline>&,                                           \
      const CudaMat<T_pipeline>&,                                           \
      const CudaMat<T_pipeline>&,                                           \
      const CudaMat<uint16_t>&,                                             \
      const CudaMat<uint16_t>&,                                             \
      const CudaMat<uint16_t>&,                                             \
      const CudaMat<uint16_t>&,                                             \
      const CudaMat<uint16_t>&,                                             \
      const CudaMat<uint16_t>&,                                             \
      CudaMat<T_compute>&,                                                  \
      CudaMat<T_compute>&,                                                  \
      CudaMat<T_compute>&,                                                  \
      const CanvasManager3&,                                                \
      float3,                                                               \
      float3,                                                               \
      float3,                                                               \
      bool,                                                                 \
      cudaStream_t);

#define INSTANTIATE_CANVAS_KERNELS(T_pipeline)                     \
  template CudaStatus launchFusedRemapToCanvasKernel3<T_pipeline>( \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      CudaMat<T_pipeline>&,                                        \
      const CanvasManager3&,                                       \
      float3,                                                      \
      float3,                                                      \
      float3,                                                      \
      bool,                                                        \
      cudaStream_t);                                               \
  template CudaStatus launchFusedRemapHardSeam3<T_pipeline>(       \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<T_pipeline>&,                                  \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<uint16_t>&,                                    \
      const CudaMat<unsigned char>&,                               \
      CudaMat<T_pipeline>&,                                        \
      const CanvasManager3&,                                       \
      float3,                                                      \
      float3,                                                      \
      float3,                                                      \
      bool,                                                        \
      cudaStream_t);

// Common instantiations
INSTANTIATE_FUSED_KERNELS(uchar3, float3)
INSTANTIATE_FUSED_KERNELS(uchar4, float4)
INSTANTIATE_FUSED_KERNELS(float3, float3)
INSTANTIATE_FUSED_KERNELS(float4, float4)

INSTANTIATE_CANVAS_KERNELS(uchar3)
INSTANTIATE_CANVAS_KERNELS(uchar4)
INSTANTIATE_CANVAS_KERNELS(float3)
INSTANTIATE_CANVAS_KERNELS(float4)

} // namespace cuda
} // namespace pano
} // namespace hm
