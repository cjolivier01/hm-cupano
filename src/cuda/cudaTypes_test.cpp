#include <gtest/gtest.h>
#include <type_traits>

#include "cupano/cuda/cudaTypes.h"

TEST(CudaTypes, BaseScalarMappings) {
  static_assert(std::is_same<BaseScalar_t<float3>, float>::value, "float3 -> float");
  static_assert(std::is_same<BaseScalar_t<uchar4>, unsigned char>::value, "uchar4 -> uchar");
  static_assert(std::is_same<BaseScalar_t<half4>, __half>::value, "half4 -> __half");
}
