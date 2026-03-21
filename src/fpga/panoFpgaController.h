#pragma once

#include "cupano/fpga/panoFpgaRuntime.h"
#include "cupano/fpga/panoFpgaTypes.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace hm {
namespace fpga {

class RegisterAccess {
 public:
  virtual ~RegisterAccess() = default;
  virtual void write32(uint32_t offset, uint32_t value) = 0;
  virtual uint32_t read32(uint32_t offset) const = 0;
};

class MmioRegisterAccess final : public RegisterAccess {
 public:
  ~MmioRegisterAccess() override;

  static std::unique_ptr<MmioRegisterAccess> Open(uint64_t physical_base, size_t bytes);

  void write32(uint32_t offset, uint32_t value) override;
  uint32_t read32(uint32_t offset) const override;

 private:
  MmioRegisterAccess(int fd, uint8_t* mapped, size_t mapped_size, size_t page_offset, size_t bytes);

  int fd_;
  uint8_t* mapped_;
  size_t mapped_size_;
  size_t page_offset_;
  size_t bytes_;
};

class PanoFpgaController : public PanoOperationExecutor {
 public:
  explicit PanoFpgaController(std::unique_ptr<RegisterAccess> regs);

  void reset();
  void clear_done();
  void program(const PanoOperation& operation);
  void start();
  bool busy() const;
  bool done() const;
  RunResult wait_for_done(std::chrono::milliseconds timeout) const;
  RunResult Run(const PanoOperation& operation, std::chrono::milliseconds timeout) override;
  RunResult run(const PanoOperation& operation, std::chrono::milliseconds timeout);

 private:
  void write_image_buffer(uint32_t addr_lo_offset, const ImageBuffer& buffer);

  std::unique_ptr<RegisterAccess> regs_;
};

} // namespace fpga
} // namespace hm
