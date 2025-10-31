#include <cupano/gpu/gpu_runtime.h>
#include <math.h>
#include <vector>

// Define CUDA block size
#define BLOCK_SIZE 16

// Simple point structures for integer (feature location) and float (flow vector) coordinates
struct Point {
  int x;
  int y;
};

struct Point2f {
  float x;
  float y;
};

// -----------------------------------------------------------------------------
// Kernel: Compute image gradients using a Sobel operator.
// -----------------------------------------------------------------------------
__global__ void computeGradients(const float* d_img, float* d_gradX, float* d_gradY, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    // Sobel for X (horizontal gradient)
    float Gx = d_img[(y - 1) * width + (x + 1)] + 2.0f * d_img[y * width + (x + 1)] + d_img[(y + 1) * width + (x + 1)] -
        d_img[(y - 1) * width + (x - 1)] - 2.0f * d_img[y * width + (x - 1)] - d_img[(y + 1) * width + (x - 1)];
    // Sobel for Y (vertical gradient)
    float Gy = d_img[(y - 1) * width + (x - 1)] + 2.0f * d_img[(y - 1) * width + x] + d_img[(y - 1) * width + (x + 1)] -
        d_img[(y + 1) * width + (x - 1)] - 2.0f * d_img[(y + 1) * width + x] - d_img[(y + 1) * width + (x + 1)];
    d_gradX[y * width + x] = Gx;
    d_gradY[y * width + x] = Gy;
  }
}

// -----------------------------------------------------------------------------
// Kernel: Compute Harris corner response for each pixel.
// Uses a simple 3x3 window and the formula: R = det(M) - k * (trace(M))^2
// -----------------------------------------------------------------------------
__global__ void computeHarrisResponse(
    const float* d_gradX,
    const float* d_gradY,
    float* d_response,
    int width,
    int height,
    float k) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
    float sumIxx = 0.0f, sumIxy = 0.0f, sumIyy = 0.0f;
    // Accumulate gradient products over a 3x3 window
    for (int j = -1; j <= 1; j++) {
      for (int i = -1; i <= 1; i++) {
        float ix = d_gradX[(y + j) * width + (x + i)];
        float iy = d_gradY[(y + j) * width + (x + i)];
        sumIxx += ix * ix;
        sumIxy += ix * iy;
        sumIyy += iy * iy;
      }
    }
    float det = sumIxx * sumIyy - sumIxy * sumIxy;
    float trace = sumIxx + sumIyy;
    d_response[y * width + x] = det - k * trace * trace;
  }
}

// -----------------------------------------------------------------------------
// Kernel: Sparse Lucas–Kanade Optical Flow
// For each feature point, iteratively refine a displacement (dx, dy)
// within a window (using a fixed 10-iteration scheme).
// -----------------------------------------------------------------------------
__global__ void lucasKanadeOpticalFlowKernel(
    const float* d_prev,
    const float* d_next,
    const Point* d_features,
    Point2f* d_flow,
    int width,
    int height,
    int window_size,
    int num_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_features)
    return;

  Point pt = d_features[idx]; // initial feature location in previous frame
  float dx = 0.0f, dy = 0.0f;
  const int half = window_size / 2;
  // Iterative refinement (10 iterations)
  for (int iter = 0; iter < 10; iter++) {
    float sumIx2 = 0.0f, sumIy2 = 0.0f, sumIxIy = 0.0f;
    float sumItIx = 0.0f, sumItIy = 0.0f;
    // Loop over the window around the feature
    for (int j = -half; j <= half; j++) {
      for (int i = -half; i <= half; i++) {
        int x = pt.x + i;
        int y = pt.y + j;
        int x_next = x + (int)roundf(dx);
        int y_next = y + (int)roundf(dy);
        // Skip if indices go out of bounds
        if (x < 0 || x >= width - 1 || y < 0 || y >= height - 1 || x_next < 0 || x_next >= width - 1 || y_next < 0 ||
            y_next >= height - 1)
          continue;
        // Compute spatial gradients using finite differences
        float Ix = (d_prev[y * width + (x + 1)] - d_prev[y * width + (x - 1)]) * 0.5f;
        float Iy = (d_prev[(y + 1) * width + x] - d_prev[(y - 1) * width + x]) * 0.5f;
        // Temporal gradient: difference between the two frames
        float It = d_next[y_next * width + x_next] - d_prev[y * width + x];
        sumIx2 += Ix * Ix;
        sumIy2 += Iy * Iy;
        sumIxIy += Ix * Iy;
        sumItIx += Ix * It;
        sumItIy += Iy * It;
      }
    }
    // Solve for displacement update [delta_x, delta_y] using the normal equations
    float det = sumIx2 * sumIy2 - sumIxIy * sumIxIy;
    if (fabsf(det) < 1e-6f)
      break; // singular matrix protection
    float delta_x = (-sumIy2 * sumItIx + sumIxIy * sumItIy) / det;
    float delta_y = (sumIxIy * sumItIx - sumIx2 * sumItIy) / det;
    dx += delta_x;
    dy += delta_y;
    if (sqrtf(delta_x * delta_x + delta_y * delta_y) < 0.01f)
      break;
  }
  d_flow[idx].x = dx;
  d_flow[idx].y = dy;
}

// -----------------------------------------------------------------------------
// Kernel: Warp (translate) a frame by a given displacement (dx, dy).
// This is used to “stabilize” the frame.
// -----------------------------------------------------------------------------
__global__ void warpFrameKernel(const float* d_input, float* d_output, int width, int height, float dx, float dy) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < width && y < height) {
    // Inverse mapping: find source coordinate in input frame
    int src_x = x - (int)roundf(dx);
    int src_y = y - (int)roundf(dy);
    if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height)
      d_output[y * width + x] = d_input[src_y * width + src_x];
    else
      d_output[y * width + x] = 0.0f; // fill border with black
  }
}

// -----------------------------------------------------------------------------
// Host Function: Harris Corner Detection
// Given a device image, this launches the gradient and Harris response kernels,
// then copies the response back and selects points above a threshold (with basic
// non-maximum suppression).
// -----------------------------------------------------------------------------
void harrisCornerDetection(
    const float* d_img,
    int width,
    int height,
    float k,
    float threshold,
    std::vector<Point>& corners) {
  float *d_gradX, *d_gradY, *d_response;
  cudaMalloc(&d_gradX, width * height * sizeof(float));
  cudaMalloc(&d_gradY, width * height * sizeof(float));
  cudaMalloc(&d_response, width * height * sizeof(float));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  computeGradients<<<grid, block>>>(d_img, d_gradX, d_gradY, width, height);
  cudaDeviceSynchronize();
  computeHarrisResponse<<<grid, block>>>(d_gradX, d_gradY, d_response, width, height, k);
  cudaDeviceSynchronize();

  float* h_response = new float[width * height];
  cudaMemcpy(h_response, d_response, width * height * sizeof(float), cudaMemcpyDeviceToHost);

  // Simple thresholding and non-maximum suppression
  for (int y = 1; y < height - 1; y++) {
    for (int x = 1; x < width - 1; x++) {
      float r = h_response[y * width + x];
      if (r > threshold) {
        bool isMax = true;
        for (int j = -1; j <= 1 && isMax; j++) {
          for (int i = -1; i <= 1; i++) {
            if (h_response[(y + j) * width + (x + i)] > r) {
              isMax = false;
              break;
            }
          }
        }
        if (isMax)
          corners.push_back({x, y});
      }
    }
  }

  delete[] h_response;
  cudaFree(d_gradX);
  cudaFree(d_gradY);
  cudaFree(d_response);
}

// -----------------------------------------------------------------------------
// Host Function: Compute Optical Flow (Lucas–Kanade)
// Given two device frames and a set of feature points (from the previous frame),
// this launches the optical flow kernel and returns a flow vector for each feature.
// -----------------------------------------------------------------------------
void computeOpticalFlow(
    const float* d_prev,
    const float* d_next,
    const std::vector<Point>& features,
    std::vector<Point2f>& flow,
    int width,
    int height,
    int window_size) {
  int num_features = features.size();
  Point* d_features;
  Point2f* d_flow;
  cudaMalloc(&d_features, num_features * sizeof(Point));
  cudaMalloc(&d_flow, num_features * sizeof(Point2f));
  cudaMemcpy(d_features, features.data(), num_features * sizeof(Point), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocks = (num_features + threadsPerBlock - 1) / threadsPerBlock;
  lucasKanadeOpticalFlowKernel<<<blocks, threadsPerBlock>>>(
      d_prev, d_next, d_features, d_flow, width, height, window_size, num_features);
  cudaDeviceSynchronize();

  flow.resize(num_features);
  cudaMemcpy(flow.data(), d_flow, num_features * sizeof(Point2f), cudaMemcpyDeviceToHost);

  cudaFree(d_features);
  cudaFree(d_flow);
}

// -----------------------------------------------------------------------------
// Host Function: Stabilize Frame
// Computes a global translation by averaging the feature flow vectors,
// then warps the frame accordingly.
// -----------------------------------------------------------------------------
void stabilizeFrame(float* d_frame, int width, int height, const std::vector<Point2f>& flow) {
  // Estimate global motion as the average flow vector.
  float sum_dx = 0.0f, sum_dy = 0.0f;
  for (auto& f : flow) {
    sum_dx += f.x;
    sum_dy += f.y;
  }
  float avg_dx = sum_dx / flow.size();
  float avg_dy = sum_dy / flow.size();

  // Allocate temporary output buffer and apply warp (translation).
  float* d_output;
  cudaMalloc(&d_output, width * height * sizeof(float));

  dim3 block(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
  warpFrameKernel<<<grid, block>>>(d_frame, d_output, width, height, avg_dx, avg_dy);
  cudaDeviceSynchronize();

  // Copy stabilized frame back into the original device memory.
  cudaMemcpy(d_frame, d_output, width * height * sizeof(float), cudaMemcpyDeviceToDevice);
  cudaFree(d_output);
}

// -----------------------------------------------------------------------------
// Example main()
// In a real GstNvStabilize element, frame buffers would come from the pipeline.
// This example loads two consecutive frames (assumed to be grayscale),
// computes feature points, optical flow, and then warps the second frame.
// -----------------------------------------------------------------------------
#ifdef CUDA_MAIN
int main() {
  // Example image dimensions; these would match your video stream.
  const int width = 640;
  const int height = 480;
  size_t frameSize = width * height * sizeof(float);

  // Allocate host buffers for two frames.
  float* h_framePrev = new float[width * height];
  float* h_frameNext = new float[width * height];
  // [TODO] Load your frame data into h_framePrev and h_frameNext here.

  // Allocate device memory and copy frames.
  float *d_framePrev, *d_frameNext;
  cudaMalloc(&d_framePrev, frameSize);
  cudaMalloc(&d_frameNext, frameSize);
  cudaMemcpy(d_framePrev, h_framePrev, frameSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_frameNext, h_frameNext, frameSize, cudaMemcpyHostToDevice);

  // Step 1: Detect features (Harris corners) in the previous frame.
  std::vector<Point> features;
  float harrisK = 0.04f;
  float threshold = 1e6f; // This threshold may need tuning for your data.
  harrisCornerDetection(d_framePrev, width, height, harrisK, threshold, features);

  // Step 2: Compute optical flow (Lucas–Kanade) from frame N-1 to N.
  std::vector<Point2f> flow;
  int window_size = 15; // window size for optical flow
  computeOpticalFlow(d_framePrev, d_frameNext, features, flow, width, height, window_size);

  // Step 3: Estimate global motion and stabilize the current frame.
  stabilizeFrame(d_frameNext, width, height, flow);

  // At this point d_frameNext contains the stabilized frame.
  // [TODO] Pass the stabilized frame into your GStreamer pipeline or further processing.

  // Cleanup
  cudaFree(d_framePrev);
  cudaFree(d_frameNext);
  delete[] h_framePrev;
  delete[] h_frameNext;

  return 0;
}
#endif // CUDA_MAIN
