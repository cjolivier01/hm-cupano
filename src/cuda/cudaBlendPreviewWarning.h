#pragma once

#include <iostream>
#include <mutex>

namespace hm {
namespace cuda_blend_detail {

inline void warn_preview_unavailable() {
  static std::once_flag once;
  std::call_once(once, [] {
    std::cerr << "CUDA blend image preview is unavailable because the OpenCV GUI module is not linked; continuing "
                 "headless."
              << std::endl;
  });
}

} // namespace cuda_blend_detail
} // namespace hm
