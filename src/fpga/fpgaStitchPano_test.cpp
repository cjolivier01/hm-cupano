#include "cupano/fpga/fpgaStitchPano.h"

#include "cupano/fpga/panoFpgaFixedPoint.h"
#include "cupano/pano/controlMasks.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include <cstring>
#include <map>
#include <vector>

namespace hm {
namespace fpga {
namespace {

class FakeAllocator final : public PhysicalBufferAllocator {
 public:
  FpgaStatusOr<PhysicalBuffer> Allocate(LogicalBuffer id, const BufferGeometry& geometry) override {
    const uint64_t phys_addr = next_phys_;
    next_phys_ += 0x10000;
    storage_[phys_addr].resize(static_cast<size_t>(geometry.stride_bytes) * static_cast<size_t>(geometry.height), 0);
    PhysicalBuffer buffer;
    buffer.id = id;
    buffer.image.phys_addr = phys_addr;
    buffer.image.width = geometry.width;
    buffer.image.height = geometry.height;
    buffer.image.stride_bytes = geometry.stride_bytes;
    buffer.image.format = geometry.format;
    buffer.virt_addr = storage_[phys_addr].data();
    buffer.size_bytes = storage_[phys_addr].size();
    return buffer;
  }

  FpgaStatus Memset(const PhysicalBuffer& buffer, uint8_t value) override {
    auto it = storage_.find(buffer.image.phys_addr);
    if (it == storage_.end()) {
      return FpgaStatus("unknown physical buffer in memset");
    }
    std::fill(it->second.begin(), it->second.end(), value);
    return FpgaStatus::OkStatus();
  }

  FpgaStatus CopyFromHost(const PhysicalBuffer& buffer, const void* src, size_t num_bytes, size_t dst_offset) override {
    auto it = storage_.find(buffer.image.phys_addr);
    if (it == storage_.end()) {
      return FpgaStatus("unknown physical buffer in copy");
    }
    if (dst_offset + num_bytes > it->second.size()) {
      return FpgaStatus("copy exceeds buffer size");
    }
    std::memcpy(it->second.data() + dst_offset, src, num_bytes);
    return FpgaStatus::OkStatus();
  }

  const std::vector<uint8_t>& bytes(uint64_t phys_addr) const {
    return storage_.at(phys_addr);
  }

 private:
  uint64_t next_phys_{0x10000000};
  std::map<uint64_t, std::vector<uint8_t>> storage_;
};

class FakeExecutor final : public PanoOperationExecutor {
 public:
  RunResult Run(const PanoOperation& operation, std::chrono::milliseconds timeout) override {
    (void)timeout;
    operations.push_back(operation);
    return RunResult{.ok = true, .status = 0, .message = "ok"};
  }

  std::vector<PanoOperation> operations;
};

TEST(FpgaStitchPanoTest, AllocatesStaticBuffersAndExecutesPlan) {
  hm::pano::ControlMasks masks;
  masks.img1_col = cv::Mat(4, 8, CV_16U, cv::Scalar(3));
  masks.img1_row = cv::Mat(4, 8, CV_16U, cv::Scalar(4));
  masks.img2_col = cv::Mat(4, 8, CV_16U, cv::Scalar(5));
  masks.img2_row = cv::Mat(4, 8, CV_16U, cv::Scalar(6));
  masks.whole_seam_mask_image = cv::Mat(4, 12, CV_8U, cv::Scalar(1));
  masks.positions = {
      hm::pano::SpatialTiff{.xpos = 0.0f, .ypos = 0.0f},
      hm::pano::SpatialTiff{.xpos = 6.0f, .ypos = 0.0f},
  };

  FakeAllocator allocator;
  FakeExecutor executor;
  FpgaStitchPano stitch(
      /*batch_size=*/1,
      /*num_levels=*/3,
      masks,
      allocator,
      executor,
      PixelFormat::kRgba8888,
      /*quiet=*/true,
      FpgaBlendProfile::kQ9_8,
      /*overlap_padding=*/2);

  ASSERT_TRUE(stitch.status().ok()) << stitch.status().message();

  const ImageBuffer input1{
      .phys_addr = 0x20000000, .width = 8, .height = 4, .stride_bytes = 32, .format = PixelFormat::kRgba8888};
  const ImageBuffer input2{
      .phys_addr = 0x21000000, .width = 8, .height = 4, .stride_bytes = 32, .format = PixelFormat::kRgba8888};

  auto result = stitch.process(input1, input2);
  ASSERT_TRUE(result.ok()) << result.status().message();
  EXPECT_EQ(result->width, stitch.plan().canvas_width);
  EXPECT_EQ(result->height, stitch.plan().canvas_height);

  ASSERT_EQ(executor.operations.size(), 6u);
  EXPECT_EQ(executor.operations.front().opcode, PanoAccelOpcode::kRemap);
  EXPECT_EQ(executor.operations.back().opcode, PanoAccelOpcode::kCopyRoi);

  const auto mask_buffer = stitch.lookup_buffer(LogicalBuffer::kBlendMask);
  ASSERT_TRUE(mask_buffer.has_value());
  const auto& mask_bytes = allocator.bytes(mask_buffer->image.phys_addr);
  ASSERT_GE(mask_bytes.size(), sizeof(uint16_t));
  uint16_t first_mask_value = 0;
  std::memcpy(&first_mask_value, mask_bytes.data(), sizeof(uint16_t));
  EXPECT_EQ(first_mask_value, fixed::kMaskOneU0_16);
}

} // namespace
} // namespace fpga
} // namespace hm
