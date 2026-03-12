#pragma once

#include <ostream>
#include <stdexcept>
#include <string>
#include <utility>

namespace hm {
namespace fpga {

class FpgaStatus {
 public:
  FpgaStatus() : ok_(true), message_("OK") {}
  explicit FpgaStatus(std::string message) : ok_(false), message_(std::move(message)) {}

  bool ok() const {
    return ok_;
  }

  const std::string& message() const {
    return message_;
  }

  static FpgaStatus OkStatus() {
    return FpgaStatus();
  }

  void Update(const FpgaStatus& other) {
    if (ok_ && !other.ok_) {
      ok_ = false;
      message_ = other.message_;
    }
  }

  friend std::ostream& operator<<(std::ostream& os, const FpgaStatus& status) {
    os << (status.ok_ ? "OK" : "ERROR") << ": " << status.message_;
    return os;
  }

 private:
  bool ok_;
  std::string message_;
};

template <typename T>
class FpgaStatusOr {
 public:
  FpgaStatusOr(const T& value) : status_(FpgaStatus::OkStatus()), value_(value), has_value_(true) {}
  FpgaStatusOr(T&& value) : status_(FpgaStatus::OkStatus()), value_(std::move(value)), has_value_(true) {}
  FpgaStatusOr(const FpgaStatus& status) : status_(status), has_value_(false) {
    if (status.ok()) {
      throw std::invalid_argument("FpgaStatusOr cannot hold an OK status without a value");
    }
  }

  bool ok() const {
    return status_.ok();
  }

  const FpgaStatus& status() const {
    return status_;
  }

  const T& ValueOrDie() const {
    if (!ok()) {
      throw std::runtime_error("FpgaStatusOr accessed without a value: " + status_.message());
    }
    return value_;
  }

  T& ValueOrDie() {
    if (!ok()) {
      throw std::runtime_error("FpgaStatusOr accessed without a value: " + status_.message());
    }
    return value_;
  }

  T ConsumeValueOrDie() {
    if (!ok()) {
      throw std::runtime_error("FpgaStatusOr consumed without a value: " + status_.message());
    }
    has_value_ = false;
    return std::move(value_);
  }

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
  FpgaStatus status_;
  T value_{};
  bool has_value_{false};
};

#define FPGA_RETURN_IF_ERROR(expr)           \
  do {                                       \
    ::hm::fpga::FpgaStatus _status = (expr); \
    if (!_status.ok()) {                     \
      return _status;                        \
    }                                        \
  } while (0)

#define FPGA_ASSIGN_OR_RETURN(lhs, rexpr) \
  do {                                    \
    auto _result = (rexpr);               \
    if (!_result.ok()) {                  \
      return _result.status();            \
    }                                     \
    lhs = _result.ConsumeValueOrDie();    \
  } while (0)

} // namespace fpga
} // namespace hm
