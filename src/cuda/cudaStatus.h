#pragma once

#include <cuda_runtime.h>
#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

//------------------------------------------------------------------------------
// CudaStatus Class
//------------------------------------------------------------------------------
class CudaStatus {
 public:
  // Default constructor produces an OK status.
  CudaStatus() : code_(cudaSuccess), message_("OK") {}

  // Constructor that takes only a cudaError_t code.
  CudaStatus(cudaError_t code) : code_(code) {
    if (code_ != cudaSuccess) {
      message_ = std::string(cudaGetErrorString(code_));
    } else {
      message_ = "OK";
    }
  }

  // Constructs an error status with a cudaError_t code and an additional message.
  CudaStatus(cudaError_t code, std::string message) : code_(code) {
    if (code_ != cudaSuccess) {
      message_ = std::string(cudaGetErrorString(code_)) + ": " + std::move(message);
    } else {
      message_ = std::move(message);
    }
  }

  // Returns true if the status is OK.
  bool ok() const {
    return code_ == cudaSuccess;
  }

  // Returns the cuda error code.
  cudaError_t code() const {
    return code_;
  }

  // Returns a descriptive error message.
  const std::string& message() const {
    return message_;
  }

  // Convenience function for an OK status.
  static CudaStatus OK() {
    return CudaStatus();
  }

  void Update(const CudaStatus& o) {
    if (ok() && !o.ok()) {
      code_ = o.code();
      message_ = o.message();
    }
  }

  // Overload the ostream << operator.
  friend std::ostream& operator<<(std::ostream& os, const CudaStatus& status) {
    os << "Code: " << status.code_ << ", Message: " << status.message_;
    return os;
  }

 private:
  cudaError_t code_;
  std::string message_;
};

//------------------------------------------------------------------------------
// CudaStatusOr<T> Template
//------------------------------------------------------------------------------
template <typename T>
class CudaStatusOr {
 public:
  // Construct a success value.
  CudaStatusOr(const T& value) : status_(CudaStatus::OK()), value_(value), has_value_(true) {}

  CudaStatusOr(T&& value) : status_(CudaStatus::OK()), value_(std::move(value)), has_value_(true) {}

  // Construct from an error status.
  // (Throws if an OK status is provided.)
  CudaStatusOr(const CudaStatus& status) : status_(status), has_value_(false) {
    if (status_.ok()) {
      throw std::invalid_argument("CudaStatusOr: cannot construct OK status without a value.");
    }
  }

  CudaStatusOr(CudaStatus&& status) : status_(std::move(status)), has_value_(false) {
    if (status_.ok()) {
      throw std::invalid_argument("CudaStatusOr: cannot construct OK status without a value.");
    }
  }

  // Copy and move constructors.
  CudaStatusOr(const CudaStatusOr& other) : status_(other.status_), has_value_(other.has_value_) {
    if (has_value_) {
      value_ = other.value_;
    }
  }

  CudaStatusOr(CudaStatusOr&& other) noexcept : status_(std::move(other.status_)), has_value_(other.has_value_) {
    if (has_value_) {
      value_ = std::move(other.value_);
      other.has_value_ = false;
    }
  }

  // Assignment operators.
  CudaStatusOr& operator=(const CudaStatusOr& other) {
    if (this != &other) {
      status_ = other.status_;
      has_value_ = other.has_value_;
      if (has_value_) {
        value_ = other.value_;
      }
    }
    return *this;
  }

  CudaStatusOr& operator=(CudaStatusOr&& other) noexcept {
    if (this != &other) {
      status_ = std::move(other.status_);
      has_value_ = other.has_value_;
      if (has_value_) {
        value_ = std::move(other.value_);
        other.has_value_ = false;
      }
    }
    return *this;
  }

  // Returns true if the contained status is OK.
  bool ok() const {
    return status_.ok();
  }

  // Returns the contained status.
  const CudaStatus& status() const {
    return status_;
  }

  // Returns the contained value or throws if the status is not OK.
  const T& ValueOrDie() const {
    if (!ok()) {
      throw std::runtime_error("CudaStatusOr: accessed value when status is not OK: " + status_.message());
    }
    return value_;
  }

  T& ValueOrDie() {
    if (!ok()) {
      throw std::runtime_error("CudaStatusOr: accessed value when status is not OK: " + status_.message());
    }
    return value_;
  }

  // Consumes the value: returns it by value (moving it out) and marks the
  // StatusOr as no longer containing a valid value.
  T ConsumeValueOrDie() {
    if (!ok()) {
      throw std::runtime_error("CudaStatusOr: consumed value when status is not OK: " + status_.message());
    }
    has_value_ = false;
    return std::move(value_);
  }

  // Provide pointer-like access.
  const T& operator*() const {
    return ValueOrDie();
  }
  T& operator*() {
    return ValueOrDie();
  }
  const T* operator->() const {
    return &ValueOrDie();
  }
  T* operator->() {
    return &ValueOrDie();
  }

 private:
  CudaStatus status_;
  T value_;
  bool has_value_;
};

//------------------------------------------------------------------------------
// CUDA Status Macros
//------------------------------------------------------------------------------
//
// CUDA_RETURN_IF_ERROR(expr):
//   Evaluates the expression (which must return a CudaStatus). If the status is not OK,
//   returns it immediately from the current function.
//
#define CUDA_RETURN_IF_ERROR(expr) \
  do {                             \
    CudaStatus _status = (expr);   \
    if (!_status.ok()) {           \
      return _status;              \
    }                              \
  } while (0)

//
// CUDA_ASSIGN_OR_RETURN(lhs, rexpr):
//   Evaluates rexpr (which must produce a CudaStatusOr<T>), and if the result is OK,
//   assigns its value to lhs; otherwise, returns the error status.
//
#define CUDA_ASSIGN_OR_RETURN(lhs, rexpr)  \
  do {                                     \
    auto _result = (rexpr);                \
    if (!_result.ok()) {                   \
      return _result.status();             \
    }                                      \
    lhs = std::move(_result.ValueOrDie()); \
  } while (0)
