#include "cupano/fpga/panoFpgaController.h"

#include "cupano/fpga/panoFpgaRegs.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace hm {
namespace fpga {
namespace {

std::string status_to_message(uint32_t status) {
  if (status & reg::bit::kError) {
    return "accelerator reported an error";
  }
  if (status & reg::bit::kTimeout) {
    return "accelerator reported a timeout";
  }
  if (status & reg::bit::kDone) {
    return "completed";
  }
  if (status & reg::bit::kBusy) {
    return "busy";
  }
  return "idle";
}

} // namespace

MmioRegisterAccess::MmioRegisterAccess(int fd, uint8_t* mapped, size_t mapped_size, size_t page_offset, size_t bytes)
    : fd_(fd), mapped_(mapped), mapped_size_(mapped_size), page_offset_(page_offset), bytes_(bytes) {}

MmioRegisterAccess::~MmioRegisterAccess() {
  if (mapped_) {
    munmap(mapped_, mapped_size_);
  }
  if (fd_ >= 0) {
    close(fd_);
  }
}

std::unique_ptr<MmioRegisterAccess> MmioRegisterAccess::Open(uint64_t physical_base, size_t bytes) {
  const long page_size = sysconf(_SC_PAGE_SIZE);
  if (page_size <= 0) {
    throw std::runtime_error("sysconf(_SC_PAGE_SIZE) failed");
  }

  const uint64_t page_mask = static_cast<uint64_t>(page_size - 1);
  const uint64_t page_base = physical_base & ~page_mask;
  const size_t page_offset = static_cast<size_t>(physical_base - page_base);
  const size_t mapped_size = ((page_offset + bytes + page_size - 1) / page_size) * page_size;

  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    throw std::runtime_error("open(/dev/mem) failed: " + std::string(std::strerror(errno)));
  }

  void* mapped = mmap(nullptr, mapped_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, static_cast<off_t>(page_base));
  if (mapped == MAP_FAILED) {
    const std::string error = std::strerror(errno);
    close(fd);
    throw std::runtime_error("mmap failed: " + error);
  }

  return std::unique_ptr<MmioRegisterAccess>(
      new MmioRegisterAccess(fd, reinterpret_cast<uint8_t*>(mapped), mapped_size, page_offset, bytes));
}

void MmioRegisterAccess::write32(uint32_t offset, uint32_t value) {
  if (offset + sizeof(uint32_t) > bytes_) {
    throw std::out_of_range("MMIO write beyond mapped register bank");
  }
  *reinterpret_cast<volatile uint32_t*>(mapped_ + page_offset_ + offset) = value;
}

uint32_t MmioRegisterAccess::read32(uint32_t offset) const {
  if (offset + sizeof(uint32_t) > bytes_) {
    throw std::out_of_range("MMIO read beyond mapped register bank");
  }
  return *reinterpret_cast<volatile uint32_t*>(mapped_ + page_offset_ + offset);
}

PanoFpgaController::PanoFpgaController(std::unique_ptr<RegisterAccess> regs) : regs_(std::move(regs)) {
  if (!regs_) {
    throw std::invalid_argument("PanoFpgaController requires a register interface");
  }
}

void PanoFpgaController::reset() {
  regs_->write32(reg::kControl, reg::bit::kSoftReset);
  regs_->write32(reg::kControl, 0);
}

void PanoFpgaController::clear_done() {
  regs_->write32(reg::kControl, reg::bit::kIrqAck);
  regs_->write32(reg::kControl, 0);
}

void PanoFpgaController::write_image_buffer(uint32_t addr_lo_offset, const ImageBuffer& buffer) {
  regs_->write32(addr_lo_offset, static_cast<uint32_t>(buffer.phys_addr & 0xffffffffu));
  regs_->write32(addr_lo_offset + 0x4, static_cast<uint32_t>(buffer.phys_addr >> 32));
  regs_->write32(addr_lo_offset + 0x8, buffer.stride_bytes);
  regs_->write32(addr_lo_offset + 0xc, reg::pack_extent(buffer.width, buffer.height));
  regs_->write32(addr_lo_offset + 0x10, reg::pack_format(buffer.format));
}

void PanoFpgaController::program(const PanoOperation& operation) {
  clear_done();
  regs_->write32(reg::kOpcode, static_cast<uint32_t>(operation.opcode));
  regs_->write32(reg::kFlags, operation.flags);

  write_image_buffer(reg::kSrc0AddrLo, operation.src0);
  write_image_buffer(reg::kSrc1AddrLo, operation.src1);
  write_image_buffer(reg::kSrc2AddrLo, operation.src2);
  write_image_buffer(reg::kDestAddrLo, operation.dest);

  regs_->write32(reg::kMapXAddrLo, static_cast<uint32_t>(operation.remap.map_x_addr & 0xffffffffu));
  regs_->write32(reg::kMapXAddrHi, static_cast<uint32_t>(operation.remap.map_x_addr >> 32));
  regs_->write32(reg::kMapYAddrLo, static_cast<uint32_t>(operation.remap.map_y_addr & 0xffffffffu));
  regs_->write32(reg::kMapYAddrHi, static_cast<uint32_t>(operation.remap.map_y_addr >> 32));
  regs_->write32(reg::kRemapExtent, reg::pack_extent(operation.remap.map_width, operation.remap.map_height));
  regs_->write32(reg::kRemapOffset, reg::pack_signed_xy(operation.remap.offset_x, operation.remap.offset_y));
  regs_->write32(reg::kRemapFlags, operation.remap.no_unmapped_write ? 1u : 0u);
  regs_->write32(reg::kAdjust01, reg::pack_adjust01(operation.remap.adjust_r_q8_8, operation.remap.adjust_g_q8_8));
  regs_->write32(reg::kAdjust2, static_cast<uint16_t>(operation.remap.adjust_b_q8_8));

  regs_->write32(reg::kCopySrcXY, reg::pack_xy(operation.copy.src_x, operation.copy.src_y));
  regs_->write32(reg::kCopyDestXY, reg::pack_xy(operation.copy.dest_x, operation.copy.dest_y));
  regs_->write32(reg::kCopyExtent, reg::pack_extent(operation.copy.width, operation.copy.height));

  regs_->write32(reg::kBlendExtent, reg::pack_extent(operation.blend.width, operation.blend.height));
  regs_->write32(reg::kBlendLevels, operation.blend.num_levels);
  regs_->write32(
      reg::kBlendCfg,
      static_cast<uint32_t>(operation.blend.channels) |
          (static_cast<uint32_t>(operation.blend.fixed_point_fraction_bits) << 16) |
          (operation.blend.alpha_aware ? (1u << 31) : 0u));

  regs_->write32(reg::kPyramidLowExtent, reg::pack_extent(operation.pyramid.low_width, operation.pyramid.low_height));
  regs_->write32(
      reg::kPyramidHighExtent, reg::pack_extent(operation.pyramid.high_width, operation.pyramid.high_height));
  regs_->write32(
      reg::kPyramidCfg,
      static_cast<uint32_t>(operation.pyramid.channels) |
          (static_cast<uint32_t>(operation.pyramid.fixed_point_fraction_bits) << 16) |
          (operation.pyramid.alpha_aware ? (1u << 31) : 0u));
}

void PanoFpgaController::start() {
  regs_->write32(reg::kControl, reg::bit::kStart);
  regs_->write32(reg::kControl, 0);
}

bool PanoFpgaController::busy() const {
  return (regs_->read32(reg::kStatus) & reg::bit::kBusy) != 0;
}

bool PanoFpgaController::done() const {
  return (regs_->read32(reg::kStatus) & reg::bit::kDone) != 0;
}

RunResult PanoFpgaController::wait_for_done(std::chrono::milliseconds timeout) const {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    const uint32_t status = regs_->read32(reg::kStatus);
    if (status & reg::bit::kDone) {
      return RunResult{.ok = (status & reg::bit::kError) == 0, .status = status, .message = status_to_message(status)};
    }
    if (status & reg::bit::kError) {
      return RunResult{.ok = false, .status = status, .message = status_to_message(status)};
    }
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }

  return RunResult{
      .ok = false,
      .status = regs_->read32(reg::kStatus) | reg::bit::kTimeout,
      .message = "timed out waiting for FPGA accelerator",
  };
}

RunResult PanoFpgaController::run(const PanoOperation& operation, std::chrono::milliseconds timeout) {
  program(operation);
  start();
  return wait_for_done(timeout);
}

RunResult PanoFpgaController::Run(const PanoOperation& operation, std::chrono::milliseconds timeout) {
  return run(operation, timeout);
}

} // namespace fpga
} // namespace hm
