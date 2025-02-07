#pragma once

#include <cuda_runtime.h>

template <typename T>
void adjustImageCudaBatch(T* d_image, int batchSize, int width, int height, const float3& adjustment);
