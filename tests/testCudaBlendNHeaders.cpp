#include <gtest/gtest.h>

#include "cupano/pano/cudaPanoN.h"

// This test only verifies that the N-image header API compiles and basic
// control-paths work without requiring CUDA libraries or device.

TEST(CudaPanoNHeadersTest, HeadersCompileAndTypesExist) {
  // This test only ensures headers parse and templates are available.
  // Avoids instantiating functions that require CUDA/IO linkage.
  using pipeline_t = float4;
  using compute_t = float4;
  (void)sizeof(hm::pano::ControlMasksN);
  (void)sizeof(hm::pano::cuda::CudaStitchPanoN<pipeline_t, compute_t>);
  SUCCEED();
}
