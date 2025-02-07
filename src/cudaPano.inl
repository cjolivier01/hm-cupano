#pragma once

#include "cudaMakeFull.h"
#include "cudaPano.h"
#include "cudaRemap.h"

namespace hm {
namespace pano {
namespace cuda {

template <typename T, typename T_compute>
CudaStitchPano<T, T_compute>::CudaStitchPano(int batch_size) {}

template <typename T, typename T_compute>
CudaStatusOr<std::unique_ptr<CudaMat<T>>> CudaStitchPano<T, T_compute>::process(
    const CudaMat<T>& sampleImage1,
    const CudaMat<T>& sampleImage2,
    StitchingContext<T, T_compute>& stitch_context,
    const CanvasManager& canvas_manager,
    cudaStream_t stream,
    std::unique_ptr<CudaMat<T>>&& canvas) {
  CudaStatus cuerr;

  assert(canvas);

  auto roi_width = [](const cv::Rect2i& roi) { return roi.width; };
  // auto roi_height = [](const cv::Rect2i& roi) { return roi.height; };

  if (!stitch_context.is_hard_seam()) {
    //
    // SOFT SEAM LEFT
    //
#if 1
    //
    // Image 1
    //
    // Remap image 1 onto the canvas
    //
    cuerr = batched_remap_kernel_ex_offset(
        sampleImage1.data(),
        sampleImage1.width(),
        sampleImage1.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_1_x->data(),
        stitch_context.remap_1_y->data(),
        {0, 0, 0},
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_1_x->width(),
        stitch_context.remap_1_x->height(),
        /*offsetX=*/canvas_manager._x1,
        /*offsetY=*/canvas_manager._y1,
        stream);
    // CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SMALL(canvas);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 1 from the canvas onto the blend image
    //
    cuerr = simple_make_full_batch<BaseScalar_t<T>, BaseScalar_t<T_compute>, unsigned char>(
        // Image 1 (float image)
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        /*region_width=*/roi_width(canvas_manager.roi_blend_1),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_1)*/,
        /*channels=*/3,
        // Batch of masks (optional)
        nullptr,
        0,
        0,
        0,
        canvas_manager.roi_blend_1.x,
        0 /* we've already applied our Y offset */,
        /*destOffsetX=*/canvas_manager._remapper_1.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull1->data_raw(),
        /*d_full_masks=*/nullptr,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(stitch_context.cudaFull1);
#endif
  } else {
    //
    // HARD SEAM LEFT
    //
#if 1
    cuerr = batched_remap_kernel_ex_offset_with_dest_map(
        sampleImage1.data(),
        sampleImage1.width(),
        sampleImage1.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_1_x->data(),
        stitch_context.remap_1_y->data(),
        {0, 0, 0},
        /*this_image_index=*/
        1 /* <-- we inverted the mask at load-time to make it a weight, so image 0 is actually 1 in the mask */,
        stitch_context.cudaBlendHardSeam->data(),
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_1_x->width(),
        stitch_context.remap_1_x->height(),
        /*offsetX=*/canvas_manager._x1,
        /*offsetY=*/canvas_manager._y1,
        stream);
    // SHOW_SMALL(&sampleImage1);
    // SHOW_IMAGE(canvas);
#endif
  }
  //
  // Image 2
  //
  if (!stitch_context.is_hard_seam()) {
    //
    // SOFT SEAM RIGHT
    //
#if 1
    //
    // Remap image 2 directly onto the canvas (will overwrite the overlappign portion of image 1)
    //
    cuerr = batched_remap_kernel_ex_offset(
        sampleImage2.data(),
        sampleImage2.width(),
        sampleImage2.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_2_x->data(),
        stitch_context.remap_2_y->data(),
        {0, 0, 0},
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_2_x->width(),
        stitch_context.remap_2_x->height(),
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/canvas_manager._y2,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_SMALL(canvas);
#endif

#if 1
    //
    // Now copy the blending portion of remapped image 2 from the canvas onto the blend image
    //
    // assert(stitch_context.cudaBlendSoftSeam->height() == roi_height(canvas_manager.roi_blend_2));
    cuerr = simple_make_full_batch<BaseScalar_t<T>, BaseScalar_t<T_compute>, unsigned char>(
        // Image 1 (float image)
        canvas->data_raw(),
        canvas->width(),
        canvas->height(),
        /*region_width=*/roi_width(canvas_manager.roi_blend_2),
        /*region_height=*/stitch_context.cudaBlendSoftSeam->height() /*roi_height(canvas_manager.roi_blend_2)*/,
        /*channels=*/3,
        // Batch of masks (optional)
        nullptr,
        0,
        0,
        0,
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/canvas_manager._y2,
        /*destOffsetX=*/canvas_manager._remapper_2.xpos,
        /*destOffsetY=*/0,
        stitch_context.cudaBlendSoftSeam->width(),
        stitch_context.cudaBlendSoftSeam->height(),
        /*adjust_origin=*/false,
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.cudaFull2->data_raw(),
        /*d_full_masks=*/nullptr,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(stitch_context.cudaFull2);
#endif
  } else {
    //
    // HARD SEAM RIGHT
    //
#if 1
    assert(canvas_manager._x2 + stitch_context.remap_2_x->width() <= canvas->width());
    assert(canvas_manager._y2 + stitch_context.remap_2_x->height() <= canvas->height());
    cuerr = batched_remap_kernel_ex_offset_with_dest_map(
        sampleImage2.data(),
        sampleImage2.width(),
        sampleImage2.height(),
        canvas->data(),
        canvas->width(),
        canvas->height(),
        stitch_context.remap_2_x->data(),
        stitch_context.remap_2_y->data(),
        {0, 0, 0},
        /*this_image_index=*/
        0 /* <-- we inverted the mask at load-time to make it a weight, so image 1 is actually 0 in the mask */,
        stitch_context.cudaBlendHardSeam->data(),
        /*batchSize=*/stitch_context.batch_size(),
        stitch_context.remap_2_x->width(),
        stitch_context.remap_2_x->height(),
        /*offsetX=*/canvas_manager._x2,
        /*offsetY=*/canvas_manager._y2,
        stream);
    // SHOW_SMALL(&sampleImage2);
    // SHOW_SMALL(canvas);
    // SHOW_SMALL(stitch_context.cudaBlendHardSeam);
#endif
  }
  if (!stitch_context.is_hard_seam()) {
    CudaMat<T_compute>& cudaBlendedFull = *stitch_context.cudaFull1;
#if 1
    //
    // BLEND THE IMAGES (overlapping portions + some padding)
    //
    cuerr = cudaBatchedLaplacianBlendWithContext(
        stitch_context.cudaFull1->data_raw(),
        stitch_context.cudaFull2->data_raw(),
        stitch_context.cudaBlendSoftSeam->data_raw(),
        // Put output in full-1 memory
        cudaBlendedFull.data_raw(),
        *stitch_context.laplacian_blend_context,
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(&cudaBlendedFull);
#endif

#if 1
    //
    // Copy the blended portion (overlapping portion + some padding) onto
    // the canvas over some of the remapped image 1 and image 2
    //
    cuerr = copyRoiBatchedInterface(
        cudaBlendedFull.data(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        cudaBlendedFull.width(),
        cudaBlendedFull.height(),
        0,
        0,
        canvas->data(),
        canvas->width(),
        canvas->height(),
        /*offsetX=*/canvas_manager._x2 - canvas_manager.overlap_padding(),
        /*offsetY=*/0,
        /*channels=*/1, // <-- 1 when using stuff like float3
        /*batchSize=*/stitch_context.batch_size(),
        stream);
    CUDA_RETURN_IF_ERROR(cuerr);
    // SHOW_IMAGE(canvas);
#endif
  }
  return std::move(canvas);
}

} // namespace cuda
} // namespace pano
} // namespace hm
