#pragma once

#include "src/cuda/cudaBlend.h"
#include "src/cuda/cudaBlend3.h"

template <typename T>
inline void CudaBatchLaplacianBlendContext<T>::displayPyramids(int channels, float scale, bool wait) const {
  hm::cuda_blend_detail::warn_preview_unavailable();
  (void)channels;
  (void)scale;
  (void)wait;
}

template <typename T>
inline void CudaBatchLaplacianBlendContext3<T>::displayPyramids(int channels, float scale, bool wait) const {
  hm::cuda_blend_detail::warn_preview_unavailable();
  (void)channels;
  (void)scale;
  (void)wait;
}
