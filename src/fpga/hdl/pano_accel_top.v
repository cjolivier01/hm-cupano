module pano_accel_top (
    input wire clk,
    input wire rstn,
    input wire cfg_write_valid,
    input wire [7:0] cfg_write_addr,
    input wire [31:0] cfg_write_data,
    input wire cfg_read_valid,
    input wire [7:0] cfg_read_addr,
    output wire [31:0] cfg_read_data,
    input wire remap_done,
    input wire copy_done,
    input wire downsample_done,
    input wire laplacian_done,
    input wire blend_done,
    input wire reconstruct_done,
    output wire remap_start,
    output wire copy_start,
    output wire downsample_start,
    output wire laplacian_start,
    output wire blend_start,
    output wire reconstruct_start,
    output wire irq,
    output wire busy,
    output wire done,
    output wire error,
    output wire [159:0] src0_desc,
    output wire [159:0] src1_desc,
    output wire [159:0] src2_desc,
    output wire [159:0] dest_desc,
    output wire [287:0] remap_desc,
    output wire [95:0] copy_desc,
    output wire [127:0] blend_desc,
    output wire [95:0] pyramid_desc
);

wire start_pulse;
wire soft_reset_pulse;
wire done_ack;
wire irq_enable;
wire [31:0] opcode;
wire [31:0] flags;
wire [31:0] status_word;

assign status_word = {28'd0, error, done, busy};

pano_mmio_regs regs (
    .clk(clk),
    .rstn(rstn),
    .cfg_write_valid(cfg_write_valid),
    .cfg_write_addr(cfg_write_addr),
    .cfg_write_data(cfg_write_data),
    .cfg_read_valid(cfg_read_valid),
    .cfg_read_addr(cfg_read_addr),
    .cfg_read_data(cfg_read_data),
    .status_word(status_word),
    .start_pulse(start_pulse),
    .soft_reset_pulse(soft_reset_pulse),
    .done_ack(done_ack),
    .irq_enable(irq_enable),
    .opcode(opcode),
    .flags(flags),
    .src0_desc(src0_desc),
    .src1_desc(src1_desc),
    .src2_desc(src2_desc),
    .dest_desc(dest_desc),
    .remap_desc(remap_desc),
    .copy_desc(copy_desc),
    .blend_desc(blend_desc),
    .pyramid_desc(pyramid_desc)
);

pano_kernel_dispatch dispatch (
    .clk(clk),
    .rstn(rstn),
    .soft_reset_pulse(soft_reset_pulse),
    .start_pulse(start_pulse),
    .done_ack(done_ack),
    .opcode(opcode),
    .irq_enable(irq_enable),
    .remap_done(remap_done),
    .copy_done(copy_done),
    .downsample_done(downsample_done),
    .laplacian_done(laplacian_done),
    .blend_done(blend_done),
    .reconstruct_done(reconstruct_done),
    .remap_start(remap_start),
    .copy_start(copy_start),
    .downsample_start(downsample_start),
    .laplacian_start(laplacian_start),
    .blend_start(blend_start),
    .reconstruct_start(reconstruct_start),
    .busy(busy),
    .done(done),
    .error(error),
    .irq(irq)
);

endmodule
