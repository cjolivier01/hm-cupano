#include "cupano/fpga/panoFpgaPlan.h"

#include "cupano/pano/controlMasks.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace hm {
namespace fpga {
namespace {

TEST(PanoFpgaPlanTest, BuildsSoftSeamPlanFromControlMasks) {
  hm::pano::ControlMasks masks;
  masks.img1_col = cv::Mat(480, 640, CV_16U, cv::Scalar(0));
  masks.img1_row = cv::Mat(480, 640, CV_16U, cv::Scalar(0));
  masks.img2_col = cv::Mat(480, 640, CV_16U, cv::Scalar(0));
  masks.img2_row = cv::Mat(480, 640, CV_16U, cv::Scalar(0));
  masks.whole_seam_mask_image = cv::Mat(480, 1024, CV_8U, cv::Scalar(255));
  masks.positions = {
      hm::pano::SpatialTiff{.xpos = 0.0f, .ypos = 0.0f},
      hm::pano::SpatialTiff{.xpos = 512.0f, .ypos = 0.0f},
  };

  const StitchPlan plan = BuildTwoCameraSoftSeamPlan(masks, 5, PixelFormat::kRgb888);

  ASSERT_EQ(plan.canvas_width, 1152u);
  ASSERT_EQ(plan.canvas_height, 480u);
  ASSERT_EQ(plan.stages.size(), 6u);
  EXPECT_EQ(plan.stages[0].opcode, PanoAccelOpcode::kRemap);
  EXPECT_EQ(plan.stages[1].opcode, PanoAccelOpcode::kCopyRoi);
  EXPECT_EQ(plan.stages[4].opcode, PanoAccelOpcode::kPyramidBlend);
  EXPECT_EQ(plan.stages[1].copy.width, 256u);
  EXPECT_EQ(plan.stages[1].copy.dest_x, 0u);
  EXPECT_EQ(plan.stages[3].copy.width, 256u);
  EXPECT_EQ(plan.stages[3].copy.dest_x, 128u);
  EXPECT_EQ(plan.stages[4].blend.width, 384u);
  EXPECT_EQ(plan.stages[5].copy.dest_x, 384u);
}

TEST(PanoFpgaPlanTest, BindsLogicalBuffersToPhysicalAddresses) {
  StitchPlan plan;
  plan.buffers = {
      BufferGeometry{
          .id = LogicalBuffer::kInput1,
          .width = 640,
          .height = 480,
          .stride_bytes = 1920,
          .format = PixelFormat::kRgb888},
      BufferGeometry{
          .id = LogicalBuffer::kCanvas,
          .width = 1152,
          .height = 480,
          .stride_bytes = 3456,
          .format = PixelFormat::kRgb888},
      BufferGeometry{
          .id = LogicalBuffer::kRemap1X,
          .width = 640,
          .height = 480,
          .stride_bytes = 1280,
          .format = PixelFormat::kGray16},
      BufferGeometry{
          .id = LogicalBuffer::kRemap1Y,
          .width = 640,
          .height = 480,
          .stride_bytes = 1280,
          .format = PixelFormat::kGray16},
  };
  plan.stages = {
      PlannedStage{
          .opcode = PanoAccelOpcode::kRemap,
          .src0 = LogicalBuffer::kInput1,
          .dest = LogicalBuffer::kCanvas,
          .map_x = LogicalBuffer::kRemap1X,
          .map_y = LogicalBuffer::kRemap1Y,
          .remap = RemapConfig{.map_width = 640, .map_height = 480},
      },
  };

  const std::vector<PanoOperation> bound = BindPlan(
      plan,
      {
          BufferBinding{.id = LogicalBuffer::kInput1, .phys_addr = 0x1000},
          BufferBinding{.id = LogicalBuffer::kCanvas, .phys_addr = 0x2000},
          BufferBinding{.id = LogicalBuffer::kRemap1X, .phys_addr = 0x3000},
          BufferBinding{.id = LogicalBuffer::kRemap1Y, .phys_addr = 0x4000},
      });

  ASSERT_EQ(bound.size(), 1u);
  EXPECT_EQ(bound[0].src0.phys_addr, 0x1000u);
  EXPECT_EQ(bound[0].dest.phys_addr, 0x2000u);
  EXPECT_EQ(bound[0].remap.map_x_addr, 0x3000u);
  EXPECT_EQ(bound[0].remap.map_y_addr, 0x4000u);
}

} // namespace
} // namespace fpga
} // namespace hm
