module pano_mmio_regs (
    input wire clk,
    input wire rstn,
    input wire cfg_write_valid,
    input wire [7:0] cfg_write_addr,
    input wire [31:0] cfg_write_data,
    input wire cfg_read_valid,
    input wire [7:0] cfg_read_addr,
    output reg [31:0] cfg_read_data,
    input wire [31:0] status_word,
    output reg start_pulse,
    output reg soft_reset_pulse,
    output reg done_ack,
    output reg irq_enable,
    output reg [31:0] opcode,
    output reg [31:0] flags,
    output reg [159:0] src0_desc,
    output reg [159:0] src1_desc,
    output reg [159:0] src2_desc,
    output reg [159:0] dest_desc,
    output reg [287:0] remap_desc,
    output reg [95:0] copy_desc,
    output reg [127:0] blend_desc,
    output reg [95:0] pyramid_desc
);

localparam REG_CONTROL = 8'h00;
localparam REG_STATUS = 8'h04;
localparam REG_IRQ_ENABLE = 8'h08;
localparam REG_OPCODE = 8'h0c;
localparam REG_FLAGS = 8'h10;
localparam REG_SRC0_ADDR_LO = 8'h20;
localparam REG_SRC0_ADDR_HI = 8'h24;
localparam REG_SRC0_STRIDE = 8'h28;
localparam REG_SRC0_EXTENT = 8'h2c;
localparam REG_SRC0_FORMAT = 8'h30;
localparam REG_SRC1_ADDR_LO = 8'h40;
localparam REG_SRC1_ADDR_HI = 8'h44;
localparam REG_SRC1_STRIDE = 8'h48;
localparam REG_SRC1_EXTENT = 8'h4c;
localparam REG_SRC1_FORMAT = 8'h50;
localparam REG_SRC2_ADDR_LO = 8'h60;
localparam REG_SRC2_ADDR_HI = 8'h64;
localparam REG_SRC2_STRIDE = 8'h68;
localparam REG_SRC2_EXTENT = 8'h6c;
localparam REG_SRC2_FORMAT = 8'h70;
localparam REG_DEST_ADDR_LO = 8'h80;
localparam REG_DEST_ADDR_HI = 8'h84;
localparam REG_DEST_STRIDE = 8'h88;
localparam REG_DEST_EXTENT = 8'h8c;
localparam REG_DEST_FORMAT = 8'h90;
localparam REG_MAPX_ADDR_LO = 8'ha0;
localparam REG_MAPX_ADDR_HI = 8'ha4;
localparam REG_MAPY_ADDR_LO = 8'ha8;
localparam REG_MAPY_ADDR_HI = 8'hac;
localparam REG_REMAP_EXTENT = 8'hb0;
localparam REG_REMAP_OFFSET = 8'hb4;
localparam REG_REMAP_FLAGS = 8'hb8;
localparam REG_ADJUST01 = 8'hbc;
localparam REG_ADJUST2 = 8'hc0;
localparam REG_COPY_SRC = 8'hc4;
localparam REG_COPY_DEST = 8'hc8;
localparam REG_COPY_EXTENT = 8'hcc;
localparam REG_BLEND_EXTENT = 8'hd0;
localparam REG_BLEND_LEVELS = 8'hd4;
localparam REG_BLEND_CFG = 8'hd8;
localparam REG_PYRAMID_LOW = 8'hdc;
localparam REG_PYRAMID_HIGH = 8'he0;
localparam REG_PYRAMID_CFG = 8'he4;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        start_pulse <= 1'b0;
        soft_reset_pulse <= 1'b0;
        done_ack <= 1'b0;
        irq_enable <= 1'b0;
        opcode <= 32'd0;
        flags <= 32'd0;
        src0_desc <= 160'd0;
        src1_desc <= 160'd0;
        src2_desc <= 160'd0;
        dest_desc <= 160'd0;
        remap_desc <= 288'd0;
        copy_desc <= 96'd0;
        blend_desc <= 128'd0;
        pyramid_desc <= 96'd0;
    end else begin
        start_pulse <= 1'b0;
        soft_reset_pulse <= 1'b0;
        done_ack <= 1'b0;

        if (cfg_write_valid) begin
            case (cfg_write_addr)
                REG_CONTROL: begin
                    start_pulse <= cfg_write_data[0];
                    soft_reset_pulse <= cfg_write_data[1];
                    done_ack <= cfg_write_data[2];
                end
                REG_IRQ_ENABLE: irq_enable <= cfg_write_data[0];
                REG_OPCODE: opcode <= cfg_write_data;
                REG_FLAGS: flags <= cfg_write_data;
                REG_SRC0_ADDR_LO: src0_desc[31:0] <= cfg_write_data;
                REG_SRC0_ADDR_HI: src0_desc[63:32] <= cfg_write_data;
                REG_SRC0_STRIDE: src0_desc[95:64] <= cfg_write_data;
                REG_SRC0_EXTENT: src0_desc[127:96] <= cfg_write_data;
                REG_SRC0_FORMAT: src0_desc[159:128] <= cfg_write_data;
                REG_SRC1_ADDR_LO: src1_desc[31:0] <= cfg_write_data;
                REG_SRC1_ADDR_HI: src1_desc[63:32] <= cfg_write_data;
                REG_SRC1_STRIDE: src1_desc[95:64] <= cfg_write_data;
                REG_SRC1_EXTENT: src1_desc[127:96] <= cfg_write_data;
                REG_SRC1_FORMAT: src1_desc[159:128] <= cfg_write_data;
                REG_SRC2_ADDR_LO: src2_desc[31:0] <= cfg_write_data;
                REG_SRC2_ADDR_HI: src2_desc[63:32] <= cfg_write_data;
                REG_SRC2_STRIDE: src2_desc[95:64] <= cfg_write_data;
                REG_SRC2_EXTENT: src2_desc[127:96] <= cfg_write_data;
                REG_SRC2_FORMAT: src2_desc[159:128] <= cfg_write_data;
                REG_DEST_ADDR_LO: dest_desc[31:0] <= cfg_write_data;
                REG_DEST_ADDR_HI: dest_desc[63:32] <= cfg_write_data;
                REG_DEST_STRIDE: dest_desc[95:64] <= cfg_write_data;
                REG_DEST_EXTENT: dest_desc[127:96] <= cfg_write_data;
                REG_DEST_FORMAT: dest_desc[159:128] <= cfg_write_data;
                REG_MAPX_ADDR_LO: remap_desc[31:0] <= cfg_write_data;
                REG_MAPX_ADDR_HI: remap_desc[63:32] <= cfg_write_data;
                REG_MAPY_ADDR_LO: remap_desc[95:64] <= cfg_write_data;
                REG_MAPY_ADDR_HI: remap_desc[127:96] <= cfg_write_data;
                REG_REMAP_EXTENT: remap_desc[159:128] <= cfg_write_data;
                REG_REMAP_OFFSET: remap_desc[191:160] <= cfg_write_data;
                REG_REMAP_FLAGS: remap_desc[223:192] <= cfg_write_data;
                REG_ADJUST01: remap_desc[255:224] <= cfg_write_data;
                REG_ADJUST2: remap_desc[287:256] <= cfg_write_data;
                REG_COPY_SRC: copy_desc[31:0] <= cfg_write_data;
                REG_COPY_DEST: copy_desc[63:32] <= cfg_write_data;
                REG_COPY_EXTENT: copy_desc[95:64] <= cfg_write_data;
                REG_BLEND_EXTENT: blend_desc[31:0] <= cfg_write_data;
                REG_BLEND_LEVELS: blend_desc[63:32] <= cfg_write_data;
                REG_BLEND_CFG: blend_desc[95:64] <= cfg_write_data;
                REG_PYRAMID_LOW: pyramid_desc[31:0] <= cfg_write_data;
                REG_PYRAMID_HIGH: pyramid_desc[63:32] <= cfg_write_data;
                REG_PYRAMID_CFG: pyramid_desc[95:64] <= cfg_write_data;
                default: begin end
            endcase
        end
    end
end

always @(*) begin
    cfg_read_data = 32'd0;
    if (cfg_read_valid) begin
        case (cfg_read_addr)
            REG_STATUS: cfg_read_data = status_word;
            REG_IRQ_ENABLE: cfg_read_data = {31'd0, irq_enable};
            REG_OPCODE: cfg_read_data = opcode;
            REG_FLAGS: cfg_read_data = flags;
            REG_SRC0_ADDR_LO: cfg_read_data = src0_desc[31:0];
            REG_SRC0_ADDR_HI: cfg_read_data = src0_desc[63:32];
            REG_SRC0_STRIDE: cfg_read_data = src0_desc[95:64];
            REG_SRC0_EXTENT: cfg_read_data = src0_desc[127:96];
            REG_SRC0_FORMAT: cfg_read_data = src0_desc[159:128];
            REG_SRC1_ADDR_LO: cfg_read_data = src1_desc[31:0];
            REG_SRC1_ADDR_HI: cfg_read_data = src1_desc[63:32];
            REG_SRC1_STRIDE: cfg_read_data = src1_desc[95:64];
            REG_SRC1_EXTENT: cfg_read_data = src1_desc[127:96];
            REG_SRC1_FORMAT: cfg_read_data = src1_desc[159:128];
            REG_SRC2_ADDR_LO: cfg_read_data = src2_desc[31:0];
            REG_SRC2_ADDR_HI: cfg_read_data = src2_desc[63:32];
            REG_SRC2_STRIDE: cfg_read_data = src2_desc[95:64];
            REG_SRC2_EXTENT: cfg_read_data = src2_desc[127:96];
            REG_SRC2_FORMAT: cfg_read_data = src2_desc[159:128];
            REG_DEST_ADDR_LO: cfg_read_data = dest_desc[31:0];
            REG_DEST_ADDR_HI: cfg_read_data = dest_desc[63:32];
            REG_DEST_STRIDE: cfg_read_data = dest_desc[95:64];
            REG_DEST_EXTENT: cfg_read_data = dest_desc[127:96];
            REG_DEST_FORMAT: cfg_read_data = dest_desc[159:128];
            REG_MAPX_ADDR_LO: cfg_read_data = remap_desc[31:0];
            REG_MAPX_ADDR_HI: cfg_read_data = remap_desc[63:32];
            REG_MAPY_ADDR_LO: cfg_read_data = remap_desc[95:64];
            REG_MAPY_ADDR_HI: cfg_read_data = remap_desc[127:96];
            REG_REMAP_EXTENT: cfg_read_data = remap_desc[159:128];
            REG_REMAP_OFFSET: cfg_read_data = remap_desc[191:160];
            REG_REMAP_FLAGS: cfg_read_data = remap_desc[223:192];
            REG_ADJUST01: cfg_read_data = remap_desc[255:224];
            REG_ADJUST2: cfg_read_data = remap_desc[287:256];
            REG_COPY_SRC: cfg_read_data = copy_desc[31:0];
            REG_COPY_DEST: cfg_read_data = copy_desc[63:32];
            REG_COPY_EXTENT: cfg_read_data = copy_desc[95:64];
            REG_BLEND_EXTENT: cfg_read_data = blend_desc[31:0];
            REG_BLEND_LEVELS: cfg_read_data = blend_desc[63:32];
            REG_BLEND_CFG: cfg_read_data = blend_desc[95:64];
            REG_PYRAMID_LOW: cfg_read_data = pyramid_desc[31:0];
            REG_PYRAMID_HIGH: cfg_read_data = pyramid_desc[63:32];
            REG_PYRAMID_CFG: cfg_read_data = pyramid_desc[95:64];
            default: cfg_read_data = 32'd0;
        endcase
    end
end

endmodule
