#pragma once

#include "cupano/fpga/panoFpgaStatus.h"
#include "cupano/fpga/panoFpgaTypes.h"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>

namespace hm {
namespace fpga {

struct PhysicalBuffer {
  LogicalBuffer id{LogicalBuffer::kInput1};
  ImageBuffer image;
  void* virt_addr{nullptr};
  size_t size_bytes{0};
};

class PhysicalBufferAllocator {
 public:
  virtual ~PhysicalBufferAllocator() = default;
  virtual FpgaStatusOr<PhysicalBuffer> Allocate(LogicalBuffer id, const BufferGeometry& geometry) = 0;
  virtual FpgaStatus Memset(const PhysicalBuffer& buffer, uint8_t value) = 0;
  virtual FpgaStatus CopyFromHost(
      const PhysicalBuffer& buffer,
      const void* src,
      size_t num_bytes,
      size_t dst_offset) = 0;
};

struct RunResult {
  bool ok{false};
  uint32_t status{0};
  std::string message;
};

class PanoOperationExecutor {
 public:
  virtual ~PanoOperationExecutor() = default;
  virtual RunResult Run(const PanoOperation& operation, std::chrono::milliseconds timeout) = 0;
};

} // namespace fpga
} // namespace hm
