#include "cupano/fpga/panoFpgaController.h"
#include "cupano/fpga/panoFpgaRegs.h"

#include <gtest/gtest.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace hm {
namespace fpga {
namespace {

class FakeRegisterAccess final : public RegisterAccess {
 public:
  void write32(uint32_t offset, uint32_t value) override {
    regs[offset] = value;
  }

  uint32_t read32(uint32_t offset) const override {
    if (offset == reg::kStatus && !status_sequence.empty()) {
      const size_t index = std::min(read_count++, status_sequence.size() - 1);
      return status_sequence[index];
    }
    const auto it = regs.find(offset);
    return it == regs.end() ? 0u : it->second;
  }

  mutable std::unordered_map<uint32_t, uint32_t> regs;
  mutable size_t read_count{0};
  std::vector<uint32_t> status_sequence;
};

TEST(PanoFpgaControllerTest, ProgramsRemapRegisters) {
  auto fake = std::make_unique<FakeRegisterAccess>();
  FakeRegisterAccess* raw = fake.get();
  PanoFpgaController controller(std::move(fake));

  PanoOperation op;
  op.opcode = PanoAccelOpcode::kRemap;
  op.src0 = ImageBuffer{
      .phys_addr = 0x123456789abcdef0ULL,
      .width = 640,
      .height = 480,
      .stride_bytes = 1920,
      .format = PixelFormat::kRgb888};
  op.dest = ImageBuffer{
      .phys_addr = 0x200000000ULL, .width = 1152, .height = 480, .stride_bytes = 3456, .format = PixelFormat::kRgb888};
  op.remap = RemapConfig{
      .map_x_addr = 0x310000000ULL,
      .map_y_addr = 0x320000000ULL,
      .map_width = 640,
      .map_height = 480,
      .offset_x = 384,
      .offset_y = 0,
      .no_unmapped_write = false,
      .adjust_r_q8_8 = -16,
      .adjust_g_q8_8 = 8,
      .adjust_b_q8_8 = 0,
  };

  controller.program(op);

  EXPECT_EQ(raw->regs[reg::kOpcode], static_cast<uint32_t>(PanoAccelOpcode::kRemap));
  EXPECT_EQ(raw->regs[reg::kSrc0AddrLo], 0x9abcdef0u);
  EXPECT_EQ(raw->regs[reg::kSrc0AddrHi], 0x12345678u);
  EXPECT_EQ(raw->regs[reg::kSrc0Extent], reg::pack_extent(640, 480));
  EXPECT_EQ(raw->regs[reg::kDestExtent], reg::pack_extent(1152, 480));
  EXPECT_EQ(raw->regs[reg::kMapXAddrLo], 0x10000000u);
  EXPECT_EQ(raw->regs[reg::kMapXAddrHi], 0x00000003u);
  EXPECT_EQ(raw->regs[reg::kMapYAddrHi], 0x00000003u);
  EXPECT_EQ(raw->regs[reg::kRemapOffset], reg::pack_signed_xy(384, 0));
  EXPECT_EQ(raw->regs[reg::kAdjust01], reg::pack_adjust01(-16, 8));
}

TEST(PanoFpgaControllerTest, WaitForDoneReturnsAfterDoneBit) {
  auto fake = std::make_unique<FakeRegisterAccess>();
  FakeRegisterAccess* raw = fake.get();
  raw->status_sequence = {reg::bit::kBusy, reg::bit::kBusy, reg::bit::kDone};
  PanoFpgaController controller(std::move(fake));

  RunResult result = controller.wait_for_done(std::chrono::milliseconds(5));

  EXPECT_TRUE(result.ok);
  EXPECT_EQ(result.status, reg::bit::kDone);
}

} // namespace
} // namespace fpga
} // namespace hm
