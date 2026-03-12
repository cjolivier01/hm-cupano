module pano_two_image_assets_pipeline #(
    parameter ADDR_WIDTH = 32,
    parameter PIXEL_WIDTH = 32,
    parameter MAP_WIDTH = 16,
    parameter INPUT_W = 16,
    parameter INPUT_H = 8,
    parameter OVERLAP = 4,
    parameter PAD = 2
) (
    input wire clk,
    input wire rstn,
    input wire start,
    output reg busy,
    output reg done
);

localparam X2 = INPUT_W - OVERLAP;
localparam CANVAS_W = INPUT_W + X2;
localparam CANVAS_H = INPUT_H;
localparam BLEND_W = OVERLAP + (2 * PAD);
localparam BLEND_H = INPUT_H;
localparam LOW_W = BLEND_W / 2;
localparam LOW_H = BLEND_H / 2;
localparam LEFT_COPY_SRC_X = X2 - PAD;
localparam LEFT_COPY_WIDTH = OVERLAP + PAD;
localparam RIGHT_COPY_SRC_X = X2;
localparam RIGHT_COPY_WIDTH = OVERLAP + PAD;
localparam RIGHT_COPY_DEST_X = PAD;
localparam PASTE_DEST_X = X2 - PAD;

localparam LEFT_BASE = 32'h0000_0000;
localparam RIGHT_BASE = 32'h0001_0000;
localparam CANVAS_BASE = 32'h0002_0000;
localparam BLEND_LEFT_BASE = 32'h0003_0000;
localparam BLEND_RIGHT_BASE = 32'h0004_0000;
localparam BLEND_OUT_BASE = 32'h0005_0000;
localparam MAP1_X_BASE = 32'h0006_0000;
localparam MAP1_Y_BASE = 32'h0007_0000;
localparam MAP2_X_BASE = 32'h0008_0000;
localparam MAP2_Y_BASE = 32'h0009_0000;

localparam ST_IDLE = 6'd0;
localparam ST_REMAP1_CFG = 6'd1;
localparam ST_REMAP1_START = 6'd2;
localparam ST_REMAP1_WAIT = 6'd3;
localparam ST_COPY1_CFG = 6'd4;
localparam ST_COPY1_START = 6'd5;
localparam ST_COPY1_WAIT = 6'd6;
localparam ST_REMAP2_CFG = 6'd7;
localparam ST_REMAP2_START = 6'd8;
localparam ST_REMAP2_WAIT = 6'd9;
localparam ST_COPY2_CFG = 6'd10;
localparam ST_COPY2_START = 6'd11;
localparam ST_COPY2_WAIT = 6'd12;
localparam ST_DS_SETUP = 6'd13;
localparam ST_DS_WRITE = 6'd14;
localparam ST_HIGH_SETUP = 6'd15;
localparam ST_HIGH_WRITE = 6'd16;
localparam ST_LOW_BLEND_SETUP = 6'd17;
localparam ST_LOW_BLEND_WRITE = 6'd18;
localparam ST_RECON_SETUP = 6'd19;
localparam ST_RECON_WRITE = 6'd20;
localparam ST_PASTE_CFG = 6'd21;
localparam ST_PASTE_START = 6'd22;
localparam ST_PASTE_WAIT = 6'd23;
localparam ST_FINISH = 6'd24;

reg [5:0] state;
reg [15:0] work_x;
reg [15:0] work_y;

reg [31:0] left_in_mem [0:(INPUT_W * INPUT_H) - 1];
reg [31:0] right_in_mem [0:(INPUT_W * INPUT_H) - 1];
reg [31:0] canvas_mem [0:(CANVAS_W * CANVAS_H) - 1];
reg [31:0] blend_left_mem [0:(BLEND_W * BLEND_H) - 1];
reg [31:0] blend_right_mem [0:(BLEND_W * BLEND_H) - 1];
reg [31:0] blend_out_mem [0:(BLEND_W * BLEND_H) - 1];
reg [15:0] map1_x_mem [0:(INPUT_W * INPUT_H) - 1];
reg [15:0] map1_y_mem [0:(INPUT_W * INPUT_H) - 1];
reg [15:0] map2_x_mem [0:(INPUT_W * INPUT_H) - 1];
reg [15:0] map2_y_mem [0:(INPUT_W * INPUT_H) - 1];
reg [15:0] mask_high_mem [0:(BLEND_W * BLEND_H) - 1];
reg [15:0] mask_low_mem [0:(LOW_W * LOW_H) - 1];
reg [71:0] low_left_mem [0:(LOW_W * LOW_H) - 1];
reg [71:0] low_right_mem [0:(LOW_W * LOW_H) - 1];
reg [71:0] high_blend_mem [0:(BLEND_W * BLEND_H) - 1];
reg [71:0] low_blend_mem [0:(LOW_W * LOW_H) - 1];

integer init_x;
integer init_y;
integer init_i;

function integer input_index;
    input integer x;
    input integer y;
    begin
        input_index = (y * INPUT_W) + x;
    end
endfunction

function integer canvas_index;
    input integer x;
    input integer y;
    begin
        canvas_index = (y * CANVAS_W) + x;
    end
endfunction

function integer blend_index;
    input integer x;
    input integer y;
    begin
        blend_index = (y * BLEND_W) + x;
    end
endfunction

function integer low_index;
    input integer x;
    input integer y;
    begin
        low_index = (y * LOW_W) + x;
    end
endfunction

function integer clamp_low_x;
    input integer x;
    begin
        if (x < 0) begin
            clamp_low_x = 0;
        end else if (x >= LOW_W) begin
            clamp_low_x = LOW_W - 1;
        end else begin
            clamp_low_x = x;
        end
    end
endfunction

function integer clamp_low_y;
    input integer y;
    begin
        if (y < 0) begin
            clamp_low_y = 0;
        end else if (y >= LOW_H) begin
            clamp_low_y = LOW_H - 1;
        end else begin
            clamp_low_y = y;
        end
    end
endfunction

function [31:0] load_pixel32;
    input [ADDR_WIDTH-1:0] addr;
    integer idx;
    begin
        load_pixel32 = 32'd0;
        if (addr >= LEFT_BASE && addr < LEFT_BASE + (INPUT_W * INPUT_H * 4)) begin
            idx = (addr - LEFT_BASE) >> 2;
            load_pixel32 = left_in_mem[idx];
        end else if (addr >= RIGHT_BASE && addr < RIGHT_BASE + (INPUT_W * INPUT_H * 4)) begin
            idx = (addr - RIGHT_BASE) >> 2;
            load_pixel32 = right_in_mem[idx];
        end else if (addr >= CANVAS_BASE && addr < CANVAS_BASE + (CANVAS_W * CANVAS_H * 4)) begin
            idx = (addr - CANVAS_BASE) >> 2;
            load_pixel32 = canvas_mem[idx];
        end else if (addr >= BLEND_LEFT_BASE && addr < BLEND_LEFT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_LEFT_BASE) >> 2;
            load_pixel32 = blend_left_mem[idx];
        end else if (addr >= BLEND_RIGHT_BASE && addr < BLEND_RIGHT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_RIGHT_BASE) >> 2;
            load_pixel32 = blend_right_mem[idx];
        end else if (addr >= BLEND_OUT_BASE && addr < BLEND_OUT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_OUT_BASE) >> 2;
            load_pixel32 = blend_out_mem[idx];
        end
    end
endfunction

function [15:0] load_map16;
    input [ADDR_WIDTH-1:0] addr;
    integer idx;
    begin
        load_map16 = 16'd0;
        if (addr >= MAP1_X_BASE && addr < MAP1_X_BASE + (INPUT_W * INPUT_H * 2)) begin
            idx = (addr - MAP1_X_BASE) >> 1;
            load_map16 = map1_x_mem[idx];
        end else if (addr >= MAP1_Y_BASE && addr < MAP1_Y_BASE + (INPUT_W * INPUT_H * 2)) begin
            idx = (addr - MAP1_Y_BASE) >> 1;
            load_map16 = map1_y_mem[idx];
        end else if (addr >= MAP2_X_BASE && addr < MAP2_X_BASE + (INPUT_W * INPUT_H * 2)) begin
            idx = (addr - MAP2_X_BASE) >> 1;
            load_map16 = map2_x_mem[idx];
        end else if (addr >= MAP2_Y_BASE && addr < MAP2_Y_BASE + (INPUT_W * INPUT_H * 2)) begin
            idx = (addr - MAP2_Y_BASE) >> 1;
            load_map16 = map2_y_mem[idx];
        end
    end
endfunction

function [71:0] pixel32_to_qpixel;
    input [31:0] pixel;
    reg [7:0] b;
    reg [7:0] g;
    reg [7:0] r;
    reg [7:0] a;
    begin
        b = pixel[7:0];
        g = pixel[15:8];
        r = pixel[23:16];
        a = pixel[31:24];
        pixel32_to_qpixel = {{2'd0, a, 8'd0}, {2'd0, r, 8'd0}, {2'd0, g, 8'd0}, {2'd0, b, 8'd0}};
    end
endfunction

function [7:0] q_to_u8;
    input signed [17:0] value;
    integer shifted;
    begin
        shifted = value >>> 8;
        if (shifted < 0) begin
            q_to_u8 = 8'd0;
        end else if (shifted > 255) begin
            q_to_u8 = 8'd255;
        end else begin
            q_to_u8 = shifted[7:0];
        end
    end
endfunction

function [31:0] qpixel_to_pixel32;
    input [71:0] pixel;
    reg [7:0] b;
    reg [7:0] g;
    reg [7:0] r;
    reg [7:0] a;
    begin
        b = q_to_u8(pixel[17:0]);
        g = q_to_u8(pixel[35:18]);
        r = q_to_u8(pixel[53:36]);
        a = q_to_u8(pixel[71:54]);
        qpixel_to_pixel32 = {a, r, g, b};
    end
endfunction

function [71:0] get_blend_left_q;
    input integer x;
    input integer y;
    begin
        get_blend_left_q = pixel32_to_qpixel(blend_left_mem[blend_index(x, y)]);
    end
endfunction

function [71:0] get_blend_right_q;
    input integer x;
    input integer y;
    begin
        get_blend_right_q = pixel32_to_qpixel(blend_right_mem[blend_index(x, y)]);
    end
endfunction

task store_pixel32;
    input [ADDR_WIDTH-1:0] addr;
    input [31:0] data;
    integer idx;
    begin
        if (addr >= CANVAS_BASE && addr < CANVAS_BASE + (CANVAS_W * CANVAS_H * 4)) begin
            idx = (addr - CANVAS_BASE) >> 2;
            canvas_mem[idx] = data;
        end else if (addr >= BLEND_LEFT_BASE && addr < BLEND_LEFT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_LEFT_BASE) >> 2;
            blend_left_mem[idx] = data;
        end else if (addr >= BLEND_RIGHT_BASE && addr < BLEND_RIGHT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_RIGHT_BASE) >> 2;
            blend_right_mem[idx] = data;
        end else if (addr >= BLEND_OUT_BASE && addr < BLEND_OUT_BASE + (BLEND_W * BLEND_H * 4)) begin
            idx = (addr - BLEND_OUT_BASE) >> 2;
            blend_out_mem[idx] = data;
        end
    end
endtask

initial begin
    for (init_i = 0; init_i < INPUT_W * INPUT_H; init_i = init_i + 1) begin
        left_in_mem[init_i] = 32'd0;
        right_in_mem[init_i] = 32'd0;
        map1_x_mem[init_i] = 16'd0;
        map1_y_mem[init_i] = 16'd0;
        map2_x_mem[init_i] = 16'd0;
        map2_y_mem[init_i] = 16'd0;
    end
    for (init_i = 0; init_i < CANVAS_W * CANVAS_H; init_i = init_i + 1) begin
        canvas_mem[init_i] = 32'd0;
    end
    for (init_i = 0; init_i < BLEND_W * BLEND_H; init_i = init_i + 1) begin
        blend_left_mem[init_i] = 32'd0;
        blend_right_mem[init_i] = 32'd0;
        blend_out_mem[init_i] = 32'd0;
        high_blend_mem[init_i] = 72'd0;
        mask_high_mem[init_i] = 16'd0;
    end
    for (init_i = 0; init_i < LOW_W * LOW_H; init_i = init_i + 1) begin
        low_left_mem[init_i] = 72'd0;
        low_right_mem[init_i] = 72'd0;
        low_blend_mem[init_i] = 72'd0;
        mask_low_mem[init_i] = 16'd0;
    end
end

reg remap_start;
reg [ADDR_WIDTH-1:0] remap_src_base_addr;
reg [ADDR_WIDTH-1:0] remap_dest_base_addr;
reg [ADDR_WIDTH-1:0] remap_map_x_base_addr;
reg [ADDR_WIDTH-1:0] remap_map_y_base_addr;
reg [31:0] remap_src_stride_bytes;
reg [31:0] remap_dest_stride_bytes;
reg [15:0] remap_src_width;
reg [15:0] remap_src_height;
reg [15:0] remap_width_cfg;
reg [15:0] remap_height_cfg;
reg signed [15:0] remap_offset_x;
reg signed [15:0] remap_offset_y;
wire remap_mapx_req_valid;
wire remap_mapx_req_ready;
wire [ADDR_WIDTH-1:0] remap_mapx_req_addr;
reg remap_mapx_rsp_valid;
wire remap_mapx_rsp_ready;
reg [MAP_WIDTH-1:0] remap_mapx_rsp_data;
wire remap_mapy_req_valid;
wire remap_mapy_req_ready;
wire [ADDR_WIDTH-1:0] remap_mapy_req_addr;
reg remap_mapy_rsp_valid;
wire remap_mapy_rsp_ready;
reg [MAP_WIDTH-1:0] remap_mapy_rsp_data;
wire remap_src_req_valid;
wire remap_src_req_ready;
wire [ADDR_WIDTH-1:0] remap_src_req_addr;
reg remap_src_rsp_valid;
wire remap_src_rsp_ready;
reg [PIXEL_WIDTH-1:0] remap_src_rsp_data;
wire remap_wr_req_valid;
wire remap_wr_req_ready;
wire [ADDR_WIDTH-1:0] remap_wr_req_addr;
wire [PIXEL_WIDTH-1:0] remap_wr_req_data;
wire [(PIXEL_WIDTH/8)-1:0] remap_wr_req_strb;
reg remap_wr_done_valid;
wire remap_wr_done_ready;
wire remap_busy_core;
wire remap_done_core;

assign remap_mapx_req_ready = 1'b1;
assign remap_mapy_req_ready = 1'b1;
assign remap_src_req_ready = 1'b1;
assign remap_wr_req_ready = 1'b1;

zybo_pano_remap_engine #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .PIXEL_WIDTH(PIXEL_WIDTH),
    .MAP_WIDTH(MAP_WIDTH)
) remap_engine (
    .clk(clk),
    .rstn(rstn),
    .start(remap_start),
    .src_base_addr(remap_src_base_addr),
    .dest_base_addr(remap_dest_base_addr),
    .map_x_base_addr(remap_map_x_base_addr),
    .map_y_base_addr(remap_map_y_base_addr),
    .src_stride_bytes(remap_src_stride_bytes),
    .dest_stride_bytes(remap_dest_stride_bytes),
    .src_width(remap_src_width),
    .src_height(remap_src_height),
    .remap_width(remap_width_cfg),
    .remap_height(remap_height_cfg),
    .offset_x(remap_offset_x),
    .offset_y(remap_offset_y),
    .no_unmapped_write(1'b0),
    .default_pixel(32'd0),
    .mapx_req_valid(remap_mapx_req_valid),
    .mapx_req_ready(remap_mapx_req_ready),
    .mapx_req_addr(remap_mapx_req_addr),
    .mapx_rsp_valid(remap_mapx_rsp_valid),
    .mapx_rsp_ready(remap_mapx_rsp_ready),
    .mapx_rsp_data(remap_mapx_rsp_data),
    .mapy_req_valid(remap_mapy_req_valid),
    .mapy_req_ready(remap_mapy_req_ready),
    .mapy_req_addr(remap_mapy_req_addr),
    .mapy_rsp_valid(remap_mapy_rsp_valid),
    .mapy_rsp_ready(remap_mapy_rsp_ready),
    .mapy_rsp_data(remap_mapy_rsp_data),
    .src_req_valid(remap_src_req_valid),
    .src_req_ready(remap_src_req_ready),
    .src_req_addr(remap_src_req_addr),
    .src_rsp_valid(remap_src_rsp_valid),
    .src_rsp_ready(remap_src_rsp_ready),
    .src_rsp_data(remap_src_rsp_data),
    .wr_req_valid(remap_wr_req_valid),
    .wr_req_ready(remap_wr_req_ready),
    .wr_req_addr(remap_wr_req_addr),
    .wr_req_data(remap_wr_req_data),
    .wr_req_strb(remap_wr_req_strb),
    .wr_done_valid(remap_wr_done_valid),
    .wr_done_ready(remap_wr_done_ready),
    .busy(remap_busy_core),
    .done(remap_done_core)
);

reg copy_start;
reg [ADDR_WIDTH-1:0] copy_src_base_addr;
reg [ADDR_WIDTH-1:0] copy_dest_base_addr;
reg [31:0] copy_src_stride_bytes;
reg [31:0] copy_dest_stride_bytes;
reg [15:0] copy_src_x;
reg [15:0] copy_src_y;
reg [15:0] copy_dest_x;
reg [15:0] copy_dest_y;
reg [15:0] copy_width_cfg;
reg [15:0] copy_height_cfg;
wire copy_rd_req_valid;
wire copy_rd_req_ready;
wire [ADDR_WIDTH-1:0] copy_rd_req_addr;
reg copy_rd_rsp_valid;
wire copy_rd_rsp_ready;
reg [PIXEL_WIDTH-1:0] copy_rd_rsp_data;
wire copy_wr_req_valid;
wire copy_wr_req_ready;
wire [ADDR_WIDTH-1:0] copy_wr_req_addr;
wire [PIXEL_WIDTH-1:0] copy_wr_req_data;
wire [(PIXEL_WIDTH/8)-1:0] copy_wr_req_strb;
reg copy_wr_done_valid;
wire copy_wr_done_ready;
wire copy_busy_core;
wire copy_done_core;

assign copy_rd_req_ready = 1'b1;
assign copy_wr_req_ready = 1'b1;

pano_copy_roi_engine #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(PIXEL_WIDTH)
) copy_engine (
    .clk(clk),
    .rstn(rstn),
    .start(copy_start),
    .src_base_addr(copy_src_base_addr),
    .dest_base_addr(copy_dest_base_addr),
    .src_stride_bytes(copy_src_stride_bytes),
    .dest_stride_bytes(copy_dest_stride_bytes),
    .src_x(copy_src_x),
    .src_y(copy_src_y),
    .dest_x(copy_dest_x),
    .dest_y(copy_dest_y),
    .width(copy_width_cfg),
    .height(copy_height_cfg),
    .rd_req_valid(copy_rd_req_valid),
    .rd_req_ready(copy_rd_req_ready),
    .rd_req_addr(copy_rd_req_addr),
    .rd_rsp_valid(copy_rd_rsp_valid),
    .rd_rsp_ready(copy_rd_rsp_ready),
    .rd_rsp_data(copy_rd_rsp_data),
    .wr_req_valid(copy_wr_req_valid),
    .wr_req_ready(copy_wr_req_ready),
    .wr_req_addr(copy_wr_req_addr),
    .wr_req_data(copy_wr_req_data),
    .wr_req_strb(copy_wr_req_strb),
    .wr_done_valid(copy_wr_done_valid),
    .wr_done_ready(copy_wr_done_ready),
    .busy(copy_busy_core),
    .done(copy_done_core)
);

reg [71:0] ds_image1_p00;
reg [71:0] ds_image1_p01;
reg [71:0] ds_image1_p10;
reg [71:0] ds_image1_p11;
reg [71:0] ds_image2_p00;
reg [71:0] ds_image2_p01;
reg [71:0] ds_image2_p10;
reg [71:0] ds_image2_p11;
reg [15:0] ds_mask_p00;
reg [15:0] ds_mask_p01;
reg [15:0] ds_mask_p10;
reg [15:0] ds_mask_p11;
wire [71:0] ds_image1_out;
wire [71:0] ds_image2_out;
wire [15:0] ds_mask_out;

pano_downsample2x2 ds_core (
    .image1_p00(ds_image1_p00),
    .image1_p01(ds_image1_p01),
    .image1_p10(ds_image1_p10),
    .image1_p11(ds_image1_p11),
    .image2_p00(ds_image2_p00),
    .image2_p01(ds_image2_p01),
    .image2_p10(ds_image2_p10),
    .image2_p11(ds_image2_p11),
    .mask_p00(ds_mask_p00),
    .mask_p01(ds_mask_p01),
    .mask_p10(ds_mask_p10),
    .mask_p11(ds_mask_p11),
    .image1_out(ds_image1_out),
    .image2_out(ds_image2_out),
    .mask_out(ds_mask_out)
);

reg lap_x_odd;
reg lap_y_odd;
reg [71:0] lap_high_left;
reg [71:0] lap_low_left00;
reg [71:0] lap_low_left10;
reg [71:0] lap_low_left01;
reg [71:0] lap_low_left11;
reg [71:0] lap_high_right;
reg [71:0] lap_low_right00;
reg [71:0] lap_low_right10;
reg [71:0] lap_low_right01;
reg [71:0] lap_low_right11;
reg [15:0] high_mask_weight;
wire [71:0] lap_left_out;
wire [71:0] lap_right_out;
wire [71:0] high_blend_out;

pano_laplacian_core lap_left_core (
    .x_odd(lap_x_odd),
    .y_odd(lap_y_odd),
    .high_pixel(lap_high_left),
    .low00(lap_low_left00),
    .low10(lap_low_left10),
    .low01(lap_low_left01),
    .low11(lap_low_left11),
    .laplacian_pixel(lap_left_out)
);

pano_laplacian_core lap_right_core (
    .x_odd(lap_x_odd),
    .y_odd(lap_y_odd),
    .high_pixel(lap_high_right),
    .low00(lap_low_right00),
    .low10(lap_low_right10),
    .low01(lap_low_right01),
    .low11(lap_low_right11),
    .laplacian_pixel(lap_right_out)
);

pano_blend_core high_blend_core (
    .lap1_pixel(lap_left_out),
    .lap2_pixel(lap_right_out),
    .mask_weight(high_mask_weight),
    .blended_pixel(high_blend_out)
);

reg [71:0] low_blend_left_pixel;
reg [71:0] low_blend_right_pixel;
reg [15:0] low_blend_mask_weight;
wire [71:0] low_blend_out_pixel;

pano_blend_core low_blend_core (
    .lap1_pixel(low_blend_left_pixel),
    .lap2_pixel(low_blend_right_pixel),
    .mask_weight(low_blend_mask_weight),
    .blended_pixel(low_blend_out_pixel)
);

reg recon_x_odd;
reg recon_y_odd;
reg [71:0] recon_low00;
reg [71:0] recon_low10;
reg [71:0] recon_low01;
reg [71:0] recon_low11;
reg [71:0] recon_lap_pixel;
wire [71:0] recon_pixel;

pano_reconstruct_core recon_core (
    .x_odd(recon_x_odd),
    .y_odd(recon_y_odd),
    .lower00(recon_low00),
    .lower10(recon_low10),
    .lower01(recon_low01),
    .lower11(recon_low11),
    .lap_pixel(recon_lap_pixel),
    .recon_pixel(recon_pixel)
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        remap_mapx_rsp_valid <= 1'b0;
        remap_mapx_rsp_data <= 16'd0;
        remap_mapy_rsp_valid <= 1'b0;
        remap_mapy_rsp_data <= 16'd0;
        remap_src_rsp_valid <= 1'b0;
        remap_src_rsp_data <= 32'd0;
        remap_wr_done_valid <= 1'b0;
        copy_rd_rsp_valid <= 1'b0;
        copy_rd_rsp_data <= 32'd0;
        copy_wr_done_valid <= 1'b0;
    end else begin
        remap_mapx_rsp_valid <= 1'b0;
        remap_mapy_rsp_valid <= 1'b0;
        remap_src_rsp_valid <= 1'b0;
        remap_wr_done_valid <= 1'b0;
        copy_rd_rsp_valid <= 1'b0;
        copy_wr_done_valid <= 1'b0;

        if (remap_mapx_req_valid) begin
            remap_mapx_rsp_data <= load_map16(remap_mapx_req_addr);
            remap_mapx_rsp_valid <= 1'b1;
        end
        if (remap_mapy_req_valid) begin
            remap_mapy_rsp_data <= load_map16(remap_mapy_req_addr);
            remap_mapy_rsp_valid <= 1'b1;
        end
        if (remap_src_req_valid) begin
            remap_src_rsp_data <= load_pixel32(remap_src_req_addr);
            remap_src_rsp_valid <= 1'b1;
        end
        if (remap_wr_req_valid) begin
            store_pixel32(remap_wr_req_addr, remap_wr_req_data);
            remap_wr_done_valid <= 1'b1;
        end
        if (copy_rd_req_valid) begin
            copy_rd_rsp_data <= load_pixel32(copy_rd_req_addr);
            copy_rd_rsp_valid <= 1'b1;
        end
        if (copy_wr_req_valid) begin
            store_pixel32(copy_wr_req_addr, copy_wr_req_data);
            copy_wr_done_valid <= 1'b1;
        end
    end
end

task configure_remap;
    input [ADDR_WIDTH-1:0] src_base;
    input [ADDR_WIDTH-1:0] dest_base;
    input [ADDR_WIDTH-1:0] mapx_base;
    input [ADDR_WIDTH-1:0] mapy_base;
    input signed [15:0] offset_x;
    input signed [15:0] offset_y;
    begin
        remap_src_base_addr <= src_base;
        remap_dest_base_addr <= dest_base;
        remap_map_x_base_addr <= mapx_base;
        remap_map_y_base_addr <= mapy_base;
        remap_src_stride_bytes <= INPUT_W * 4;
        remap_dest_stride_bytes <= CANVAS_W * 4;
        remap_src_width <= INPUT_W;
        remap_src_height <= INPUT_H;
        remap_width_cfg <= INPUT_W;
        remap_height_cfg <= INPUT_H;
        remap_offset_x <= offset_x;
        remap_offset_y <= offset_y;
    end
endtask

task configure_copy;
    input [ADDR_WIDTH-1:0] src_base;
    input [ADDR_WIDTH-1:0] dest_base;
    input [31:0] src_stride;
    input [31:0] dest_stride;
    input [15:0] src_x_cfg;
    input [15:0] src_y_cfg;
    input [15:0] dest_x_cfg;
    input [15:0] dest_y_cfg;
    input [15:0] width_cfg;
    input [15:0] height_cfg;
    begin
        copy_src_base_addr <= src_base;
        copy_dest_base_addr <= dest_base;
        copy_src_stride_bytes <= src_stride;
        copy_dest_stride_bytes <= dest_stride;
        copy_src_x <= src_x_cfg;
        copy_src_y <= src_y_cfg;
        copy_dest_x <= dest_x_cfg;
        copy_dest_y <= dest_y_cfg;
        copy_width_cfg <= width_cfg;
        copy_height_cfg <= height_cfg;
    end
endtask

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= ST_IDLE;
        work_x <= 16'd0;
        work_y <= 16'd0;
        busy <= 1'b0;
        done <= 1'b0;
        remap_start <= 1'b0;
        copy_start <= 1'b0;
        remap_src_base_addr <= LEFT_BASE;
        remap_dest_base_addr <= CANVAS_BASE;
        remap_map_x_base_addr <= MAP1_X_BASE;
        remap_map_y_base_addr <= MAP1_Y_BASE;
        remap_src_stride_bytes <= INPUT_W * 4;
        remap_dest_stride_bytes <= CANVAS_W * 4;
        remap_src_width <= INPUT_W;
        remap_src_height <= INPUT_H;
        remap_width_cfg <= INPUT_W;
        remap_height_cfg <= INPUT_H;
        remap_offset_x <= 16'sd0;
        remap_offset_y <= 16'sd0;
        copy_src_base_addr <= CANVAS_BASE;
        copy_dest_base_addr <= BLEND_LEFT_BASE;
        copy_src_stride_bytes <= CANVAS_W * 4;
        copy_dest_stride_bytes <= BLEND_W * 4;
        copy_src_x <= 16'd0;
        copy_src_y <= 16'd0;
        copy_dest_x <= 16'd0;
        copy_dest_y <= 16'd0;
        copy_width_cfg <= 16'd0;
        copy_height_cfg <= 16'd0;
        ds_image1_p00 <= 72'd0;
        ds_image1_p01 <= 72'd0;
        ds_image1_p10 <= 72'd0;
        ds_image1_p11 <= 72'd0;
        ds_image2_p00 <= 72'd0;
        ds_image2_p01 <= 72'd0;
        ds_image2_p10 <= 72'd0;
        ds_image2_p11 <= 72'd0;
        ds_mask_p00 <= 16'd0;
        ds_mask_p01 <= 16'd0;
        ds_mask_p10 <= 16'd0;
        ds_mask_p11 <= 16'd0;
        lap_x_odd <= 1'b0;
        lap_y_odd <= 1'b0;
        lap_high_left <= 72'd0;
        lap_low_left00 <= 72'd0;
        lap_low_left10 <= 72'd0;
        lap_low_left01 <= 72'd0;
        lap_low_left11 <= 72'd0;
        lap_high_right <= 72'd0;
        lap_low_right00 <= 72'd0;
        lap_low_right10 <= 72'd0;
        lap_low_right01 <= 72'd0;
        lap_low_right11 <= 72'd0;
        high_mask_weight <= 16'd0;
        low_blend_left_pixel <= 72'd0;
        low_blend_right_pixel <= 72'd0;
        low_blend_mask_weight <= 16'd0;
        recon_x_odd <= 1'b0;
        recon_y_odd <= 1'b0;
        recon_low00 <= 72'd0;
        recon_low10 <= 72'd0;
        recon_low01 <= 72'd0;
        recon_low11 <= 72'd0;
        recon_lap_pixel <= 72'd0;
    end else begin
        remap_start <= 1'b0;
        copy_start <= 1'b0;
        done <= 1'b0;

        case (state)
            ST_IDLE: begin
                busy <= 1'b0;
                work_x <= 16'd0;
                work_y <= 16'd0;
                if (start) begin
                    busy <= 1'b1;
                    state <= ST_REMAP1_CFG;
                end
            end

            ST_REMAP1_CFG: begin
                configure_remap(LEFT_BASE, CANVAS_BASE, MAP1_X_BASE, MAP1_Y_BASE, 16'sd0, 16'sd0);
                state <= ST_REMAP1_START;
            end
            ST_REMAP1_START: begin
                remap_start <= 1'b1;
                state <= ST_REMAP1_WAIT;
            end
            ST_REMAP1_WAIT: begin
                if (remap_done_core) begin
                    state <= ST_COPY1_CFG;
                end
            end

            ST_COPY1_CFG: begin
                configure_copy(
                    CANVAS_BASE,
                    BLEND_LEFT_BASE,
                    CANVAS_W * 4,
                    BLEND_W * 4,
                    LEFT_COPY_SRC_X,
                    16'd0,
                    16'd0,
                    16'd0,
                    LEFT_COPY_WIDTH,
                    BLEND_H
                );
                state <= ST_COPY1_START;
            end
            ST_COPY1_START: begin
                copy_start <= 1'b1;
                state <= ST_COPY1_WAIT;
            end
            ST_COPY1_WAIT: begin
                if (copy_done_core) begin
                    state <= ST_REMAP2_CFG;
                end
            end

            ST_REMAP2_CFG: begin
                configure_remap(RIGHT_BASE, CANVAS_BASE, MAP2_X_BASE, MAP2_Y_BASE, X2, 16'sd0);
                state <= ST_REMAP2_START;
            end
            ST_REMAP2_START: begin
                remap_start <= 1'b1;
                state <= ST_REMAP2_WAIT;
            end
            ST_REMAP2_WAIT: begin
                if (remap_done_core) begin
                    state <= ST_COPY2_CFG;
                end
            end

            ST_COPY2_CFG: begin
                configure_copy(
                    CANVAS_BASE,
                    BLEND_RIGHT_BASE,
                    CANVAS_W * 4,
                    BLEND_W * 4,
                    RIGHT_COPY_SRC_X,
                    16'd0,
                    RIGHT_COPY_DEST_X,
                    16'd0,
                    RIGHT_COPY_WIDTH,
                    BLEND_H
                );
                state <= ST_COPY2_START;
            end
            ST_COPY2_START: begin
                copy_start <= 1'b1;
                state <= ST_COPY2_WAIT;
            end
            ST_COPY2_WAIT: begin
                if (copy_done_core) begin
                    work_x <= 16'd0;
                    work_y <= 16'd0;
                    state <= ST_DS_SETUP;
                end
            end

            ST_DS_SETUP: begin
                ds_image1_p00 <= get_blend_left_q(work_x * 2, work_y * 2);
                ds_image1_p01 <= get_blend_left_q((work_x * 2) + 1, work_y * 2);
                ds_image1_p10 <= get_blend_left_q(work_x * 2, (work_y * 2) + 1);
                ds_image1_p11 <= get_blend_left_q((work_x * 2) + 1, (work_y * 2) + 1);
                ds_image2_p00 <= get_blend_right_q(work_x * 2, work_y * 2);
                ds_image2_p01 <= get_blend_right_q((work_x * 2) + 1, work_y * 2);
                ds_image2_p10 <= get_blend_right_q(work_x * 2, (work_y * 2) + 1);
                ds_image2_p11 <= get_blend_right_q((work_x * 2) + 1, (work_y * 2) + 1);
                ds_mask_p00 <= mask_high_mem[blend_index(work_x * 2, work_y * 2)];
                ds_mask_p01 <= mask_high_mem[blend_index((work_x * 2) + 1, work_y * 2)];
                ds_mask_p10 <= mask_high_mem[blend_index(work_x * 2, (work_y * 2) + 1)];
                ds_mask_p11 <= mask_high_mem[blend_index((work_x * 2) + 1, (work_y * 2) + 1)];
                state <= ST_DS_WRITE;
            end
            ST_DS_WRITE: begin
                low_left_mem[low_index(work_x, work_y)] <= ds_image1_out;
                low_right_mem[low_index(work_x, work_y)] <= ds_image2_out;
                mask_low_mem[low_index(work_x, work_y)] <= ds_mask_out;
                if (work_x == LOW_W - 1) begin
                    work_x <= 16'd0;
                    if (work_y == LOW_H - 1) begin
                        work_y <= 16'd0;
                        state <= ST_HIGH_SETUP;
                    end else begin
                        work_y <= work_y + 16'd1;
                        state <= ST_DS_SETUP;
                    end
                end else begin
                    work_x <= work_x + 16'd1;
                    state <= ST_DS_SETUP;
                end
            end

            ST_HIGH_SETUP: begin
                lap_x_odd <= work_x[0];
                lap_y_odd <= work_y[0];
                lap_high_left <= get_blend_left_q(work_x, work_y);
                lap_low_left00 <= low_left_mem[low_index(work_x >> 1, work_y >> 1)];
                lap_low_left10 <= low_left_mem[low_index(clamp_low_x((work_x >> 1) + 1), work_y >> 1)];
                lap_low_left01 <= low_left_mem[low_index(work_x >> 1, clamp_low_y((work_y >> 1) + 1))];
                lap_low_left11 <= low_left_mem[low_index(clamp_low_x((work_x >> 1) + 1), clamp_low_y((work_y >> 1) + 1))];
                lap_high_right <= get_blend_right_q(work_x, work_y);
                lap_low_right00 <= low_right_mem[low_index(work_x >> 1, work_y >> 1)];
                lap_low_right10 <= low_right_mem[low_index(clamp_low_x((work_x >> 1) + 1), work_y >> 1)];
                lap_low_right01 <= low_right_mem[low_index(work_x >> 1, clamp_low_y((work_y >> 1) + 1))];
                lap_low_right11 <= low_right_mem[low_index(clamp_low_x((work_x >> 1) + 1), clamp_low_y((work_y >> 1) + 1))];
                high_mask_weight <= mask_high_mem[blend_index(work_x, work_y)];
                state <= ST_HIGH_WRITE;
            end
            ST_HIGH_WRITE: begin
                high_blend_mem[blend_index(work_x, work_y)] <= high_blend_out;
                if (work_x == BLEND_W - 1) begin
                    work_x <= 16'd0;
                    if (work_y == BLEND_H - 1) begin
                        work_y <= 16'd0;
                        state <= ST_LOW_BLEND_SETUP;
                    end else begin
                        work_y <= work_y + 16'd1;
                        state <= ST_HIGH_SETUP;
                    end
                end else begin
                    work_x <= work_x + 16'd1;
                    state <= ST_HIGH_SETUP;
                end
            end

            ST_LOW_BLEND_SETUP: begin
                low_blend_left_pixel <= low_left_mem[low_index(work_x, work_y)];
                low_blend_right_pixel <= low_right_mem[low_index(work_x, work_y)];
                low_blend_mask_weight <= mask_low_mem[low_index(work_x, work_y)];
                state <= ST_LOW_BLEND_WRITE;
            end
            ST_LOW_BLEND_WRITE: begin
                low_blend_mem[low_index(work_x, work_y)] <= low_blend_out_pixel;
                if (work_x == LOW_W - 1) begin
                    work_x <= 16'd0;
                    if (work_y == LOW_H - 1) begin
                        work_y <= 16'd0;
                        state <= ST_RECON_SETUP;
                    end else begin
                        work_y <= work_y + 16'd1;
                        state <= ST_LOW_BLEND_SETUP;
                    end
                end else begin
                    work_x <= work_x + 16'd1;
                    state <= ST_LOW_BLEND_SETUP;
                end
            end

            ST_RECON_SETUP: begin
                recon_x_odd <= work_x[0];
                recon_y_odd <= work_y[0];
                recon_low00 <= low_blend_mem[low_index(work_x >> 1, work_y >> 1)];
                recon_low10 <= low_blend_mem[low_index(clamp_low_x((work_x >> 1) + 1), work_y >> 1)];
                recon_low01 <= low_blend_mem[low_index(work_x >> 1, clamp_low_y((work_y >> 1) + 1))];
                recon_low11 <= low_blend_mem[low_index(clamp_low_x((work_x >> 1) + 1), clamp_low_y((work_y >> 1) + 1))];
                recon_lap_pixel <= high_blend_mem[blend_index(work_x, work_y)];
                state <= ST_RECON_WRITE;
            end
            ST_RECON_WRITE: begin
                blend_out_mem[blend_index(work_x, work_y)] <= qpixel_to_pixel32(recon_pixel);
                if (work_x == BLEND_W - 1) begin
                    work_x <= 16'd0;
                    if (work_y == BLEND_H - 1) begin
                        work_y <= 16'd0;
                        state <= ST_PASTE_CFG;
                    end else begin
                        work_y <= work_y + 16'd1;
                        state <= ST_RECON_SETUP;
                    end
                end else begin
                    work_x <= work_x + 16'd1;
                    state <= ST_RECON_SETUP;
                end
            end

            ST_PASTE_CFG: begin
                configure_copy(
                    BLEND_OUT_BASE,
                    CANVAS_BASE,
                    BLEND_W * 4,
                    CANVAS_W * 4,
                    16'd0,
                    16'd0,
                    PASTE_DEST_X,
                    16'd0,
                    BLEND_W,
                    BLEND_H
                );
                state <= ST_PASTE_START;
            end
            ST_PASTE_START: begin
                copy_start <= 1'b1;
                state <= ST_PASTE_WAIT;
            end
            ST_PASTE_WAIT: begin
                if (copy_done_core) begin
                    state <= ST_FINISH;
                end
            end

            ST_FINISH: begin
                busy <= 1'b0;
                done <= 1'b1;
                state <= ST_IDLE;
            end

            default: state <= ST_IDLE;
        endcase
    end
end

endmodule
