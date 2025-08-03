#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cassert>
#include <stdexcept>
#include <string>
#include "cupano/pano/cudaMat.h"

namespace hm {
namespace utils {

// Helper macro for OpenGL error checking
#define CHECK_GL_ERROR()                                                             \
  do {                                                                               \
    GLenum err = glGetError();                                                       \
    if (err != GL_NO_ERROR) {                                                        \
      throw std::runtime_error(std::string("OpenGL error: ") + std::to_string(err)); \
    }                                                                                \
  } while (0)

// Helper macro for CUDA error checking
#define CHECK_CUDA_ERR(call)                                                                  \
  do {                                                                                        \
    cudaError_t err = call;                                                                   \
    if (err != cudaSuccess) {                                                                 \
      throw std::runtime_error(                                                               \
          std::string("CUDA error at ") + __FILE__ + ":" + std::to_string(__LINE__) + " - " + \
          cudaGetErrorString(err));                                                           \
    }                                                                                         \
  } while (0)

class CudaGLWindow {
  // using KeyCallback = std::function<void(int key, int scancode, int action,
  // int mods)>;

 public:
  CudaGLWindow(int width, int height, int channels, const std::string& title)
      : window_(nullptr), texture_(0), cudaRes_(nullptr), width_(width), height_(height), channels_(channels) {
    if (!glfwInit()) {
      throw std::runtime_error("Failed to initialize GLFW");
    }

    window_ = glfwCreateWindow(width_, height_, title.c_str(), nullptr, nullptr);
    if (!window_) {
      glfwTerminate();
      throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window_);

    // Setup key callback forwarding
    glfwSetKeyCallback(window_, &CudaGLWindow::globalKeyCallback);

    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK) {
      glfwDestroyWindow(window_);
      glfwTerminate();
      throw std::runtime_error(std::string("GLEW init error: ") + (const char*)glewGetErrorString(glewErr));
    }

    // Create OpenGL texture
    glGenTextures(1, &texture_);
    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture_);
    CHECK_GL_ERROR();

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    /*! \brief Define a two-dimensional texture image.
     *
     *  API Signature:
     *  void glTexImage2D(GLenum target,
     *                    GLint level,
     *                    GLint internalformat,
     *                    GLsizei width,
     *                    GLsizei height,
     *                    GLint border,
     *                    GLenum format,
     *                    GLenum type,
     *                    const void* pixels);
     *
     *  - target:       The texture target, e.g., GL_TEXTURE_2D.
     *  - level:        Mipmap level-of-detail number. 0 is base image.
     *  - internalformat: Number of color components or sized internal storage format (e.g., GL_RGBA8).
     *  - width:        Width of the texture image. Must be >= 0.
     *  - height:       Height of the texture image. Must be >= 0.
     *  - border:       Width of the border. Must be 0.
     *  - format:       Format of the pixel data (e.g., GL_RGBA, GL_BGRA, GL_RED).
     *  - type:         Data type of the pixel data (e.g., GL_UNSIGNED_BYTE, GL_FLOAT).
     *  - pixels:       Pointer to image data in client memory. If nullptr, allocates storage without initializing.
     */
    if (channels_ == 4) {
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width_, height_, 0, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    } else {
      assert(channels_ == 3);
      glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width_, height_, 0, GL_BGR, GL_UNSIGNED_BYTE, nullptr);
    }

    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, 0);
    CHECK_GL_ERROR();

    // Register texture with CUDA
    CHECK_CUDA_ERR(
        cudaGraphicsGLRegisterImage(&cudaRes_, texture_, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));
  }

  ~CudaGLWindow() {
    cleanup();
  }
  bool shouldClose() const {
    return window_ && glfwWindowShouldClose(window_);
  }

  // Hook a callback for key events (key, scancode, action, mods)
  // void setKeyCallback(KeyCallback cb) {
  //   keyCallback_ = std::move(cb);
  // }

  // Check if a specific key is currently pressed
  bool isKeyPressed(int key) const {
    if (!window_) {
      throw std::runtime_error("Window has been destroyed");
    }
    // glfwGetKey returns GLFW_PRESS or GLFW_RELEASE
    return glfwGetKey(window_, key) == GLFW_PRESS;
  }

  // Render the given device image (CV_8UC4) to the window
  template <typename PIXEL_T>
  void render(const CudaSurface<PIXEL_T>& d_img, cudaStream_t stream = nullptr) {
    assert(sizeof(PIXEL_T)/sizeof(PIXEL_T::x) == channels_);
    assert((int)d_img.width == width_);
    assert((int)d_img.height == height_);

    // Map the GL texture as a CUDA array
    CHECK_CUDA_ERR(cudaGraphicsMapResources(1, &cudaRes_, 0));
    cudaArray_t arr;
    CHECK_CUDA_ERR(cudaGraphicsSubResourceGetMappedArray(&arr, cudaRes_, 0, 0));

    // Copy from GpuMat to CUDA array (device->device)
    CHECK_CUDA_ERR(cudaMemcpy2DToArray(
        arr, // destination CUDA array
        0,
        0, // x/y offset
        d_img.d_ptr, // source pointer (device)
        d_img.pitch, // source pitch
        d_img.width * sizeof(PIXEL_T), // row size in bytes
        d_img.height, // number of rows
        cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERR(cudaGraphicsUnmapResources(1, &cudaRes_, 0));

    // Draw textured quad
    glClear(GL_COLOR_BUFFER_BIT);
    CHECK_GL_ERROR();

    glEnable(GL_TEXTURE_2D);
    CHECK_GL_ERROR();

    glBindTexture(GL_TEXTURE_2D, texture_);
    CHECK_GL_ERROR();

    glBegin(GL_QUADS);
    // Vertically flip the texture coords to correct upside-down image
    glTexCoord2f(0.f, 1.f);
    glVertex2f(-1.f, -1.f);
    glTexCoord2f(1.f, 1.f);
    glVertex2f(1.f, -1.f);
    glTexCoord2f(1.f, 0.f);
    glVertex2f(1.f, 1.f);
    glTexCoord2f(0.f, 0.f);
    glVertex2f(-1.f, 1.f);
    glEnd();
    CHECK_GL_ERROR();

    glfwSwapBuffers(window_);
    CHECK_GL_ERROR();
    glfwPollEvents();
    CHECK_GL_ERROR();
  }

 private:
  // Global trampoline for GLFW key events
  static void globalKeyCallback(GLFWwindow* win, int key, int scancode, int action, int mods) {
    std::cout << "globalKeyCallback()" << std::endl;
    // if (s_instance && s_instance->keyCallback_) {
    //   s_instance->keyCallback_(key, scancode, action, mods);
    // }
  }

  void cleanup() {
    if (cudaRes_) {
      cudaGraphicsUnregisterResource(cudaRes_);
      cudaRes_ = nullptr;
    }
    if (texture_) {
      glDeleteTextures(1, &texture_);
      texture_ = 0;
    }
    if (window_) {
      glfwDestroyWindow(window_);
      window_ = nullptr;
    }
    glfwTerminate();
  }
  GLFWwindow* window_;
  GLuint texture_;
  cudaGraphicsResource* cudaRes_;
  int width_, height_;
  int channels_;
};

} // namespace utils
} // namespace hm
