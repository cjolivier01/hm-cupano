module zybo_pano_remap_engine #(
    parameter ADDR_WIDTH = 32,
    parameter PIXEL_WIDTH = 32,
    parameter MAP_WIDTH = 16,
    parameter PIXEL_BYTES = 4,
    parameter MAP_BYTES = 2
) (
    input wire clk,
    input wire rstn,
    input wire start,
    input wire [ADDR_WIDTH-1:0] src_base_addr,
    input wire [ADDR_WIDTH-1:0] dest_base_addr,
    input wire [ADDR_WIDTH-1:0] map_x_base_addr,
    input wire [ADDR_WIDTH-1:0] map_y_base_addr,
    input wire [31:0] src_stride_bytes,
    input wire [31:0] dest_stride_bytes,
    input wire [15:0] src_width,
    input wire [15:0] src_height,
    input wire [15:0] remap_width,
    input wire [15:0] remap_height,
    input wire signed [15:0] offset_x,
    input wire signed [15:0] offset_y,
    input wire no_unmapped_write,
    input wire [PIXEL_WIDTH-1:0] default_pixel,
    output reg mapx_req_valid,
    input wire mapx_req_ready,
    output reg [ADDR_WIDTH-1:0] mapx_req_addr,
    input wire mapx_rsp_valid,
    output reg mapx_rsp_ready,
    input wire [MAP_WIDTH-1:0] mapx_rsp_data,
    output reg mapy_req_valid,
    input wire mapy_req_ready,
    output reg [ADDR_WIDTH-1:0] mapy_req_addr,
    input wire mapy_rsp_valid,
    output reg mapy_rsp_ready,
    input wire [MAP_WIDTH-1:0] mapy_rsp_data,
    output wire src_req_valid,
    input wire src_req_ready,
    output wire [ADDR_WIDTH-1:0] src_req_addr,
    input wire src_rsp_valid,
    output wire src_rsp_ready,
    input wire [PIXEL_WIDTH-1:0] src_rsp_data,
    output reg wr_req_valid,
    input wire wr_req_ready,
    output reg [ADDR_WIDTH-1:0] wr_req_addr,
    output reg [PIXEL_WIDTH-1:0] wr_req_data,
    output reg [(PIXEL_WIDTH/8)-1:0] wr_req_strb,
    input wire wr_done_valid,
    output reg wr_done_ready,
    output reg busy,
    output reg done
);

localparam ST_IDLE = 3'd0;
localparam ST_FETCH_X = 3'd1;
localparam ST_WAIT_X = 3'd2;
localparam ST_FETCH_Y = 3'd3;
localparam ST_WAIT_Y = 3'd4;

reg [2:0] map_state;
reg [31:0] map_index;
reg [MAP_WIDTH-1:0] map_x_value;
reg [MAP_WIDTH-1:0] map_y_value;
reg map_valid;
wire map_ready;
wire core_busy;
wire core_done;
wire dst_valid;
wire dst_ready;
wire signed [15:0] dst_x;
wire signed [15:0] dst_y;
wire [PIXEL_WIDTH-1:0] dst_pixel;
reg write_inflight;
reg core_done_pending;

function [ADDR_WIDTH-1:0] map_addr;
    input [ADDR_WIDTH-1:0] base_addr;
    input [31:0] index;
    begin
        map_addr = base_addr + (index * MAP_BYTES);
    end
endfunction

function [ADDR_WIDTH-1:0] pixel_addr;
    input [ADDR_WIDTH-1:0] base_addr;
    input [31:0] stride_bytes;
    input signed [15:0] x;
    input signed [15:0] y;
    begin
        pixel_addr = base_addr + (y * stride_bytes) + (x * PIXEL_BYTES);
    end
endfunction

assign dst_ready = !write_inflight && wr_req_ready;
assign src_rsp_ready = 1'b1;

pano_remap_core #(
    .DATA_WIDTH(PIXEL_WIDTH),
    .COORD_WIDTH(MAP_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .STRIDE_WIDTH(16)
) core (
    .clk(clk),
    .rstn(rstn),
    .start(start),
    .remap_width(remap_width),
    .remap_height(remap_height),
    .src_width(src_width),
    .src_height(src_height),
    .src_base_addr(src_base_addr),
    .src_stride_bytes(src_stride_bytes[15:0]),
    .bytes_per_pixel(16'd4),
    .offset_x(offset_x),
    .offset_y(offset_y),
    .no_unmapped_write(no_unmapped_write),
    .default_pixel(default_pixel),
    .map_valid(map_valid),
    .map_ready(map_ready),
    .map_x(map_x_value),
    .map_y(map_y_value),
    .src_req_valid(src_req_valid),
    .src_req_ready(src_req_ready),
    .src_req_addr(src_req_addr),
    .src_rsp_valid(src_rsp_valid),
    .src_rsp_data(src_rsp_data),
    .dst_valid(dst_valid),
    .dst_ready(dst_ready),
    .dst_x(dst_x),
    .dst_y(dst_y),
    .dst_pixel(dst_pixel),
    .busy(core_busy),
    .done(core_done)
);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        map_state <= ST_IDLE;
        map_index <= 32'd0;
        map_x_value <= {MAP_WIDTH{1'b0}};
        map_y_value <= {MAP_WIDTH{1'b0}};
        map_valid <= 1'b0;
        mapx_req_valid <= 1'b0;
        mapx_req_addr <= {ADDR_WIDTH{1'b0}};
        mapx_rsp_ready <= 1'b0;
        mapy_req_valid <= 1'b0;
        mapy_req_addr <= {ADDR_WIDTH{1'b0}};
        mapy_rsp_ready <= 1'b0;
        wr_req_valid <= 1'b0;
        wr_req_addr <= {ADDR_WIDTH{1'b0}};
        wr_req_data <= {PIXEL_WIDTH{1'b0}};
        wr_req_strb <= {(PIXEL_WIDTH/8){1'b1}};
        wr_done_ready <= 1'b0;
        write_inflight <= 1'b0;
        core_done_pending <= 1'b0;
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;

        if (start && !busy) begin
            busy <= 1'b1;
            map_index <= 32'd0;
            map_state <= ST_FETCH_X;
            map_valid <= 1'b0;
            core_done_pending <= 1'b0;
        end

        case (map_state)
            ST_IDLE: begin
                map_valid <= 1'b0;
                mapx_req_valid <= 1'b0;
                mapx_rsp_ready <= 1'b0;
                mapy_req_valid <= 1'b0;
                mapy_rsp_ready <= 1'b0;
            end
            ST_FETCH_X: begin
                if (map_ready && !map_valid) begin
                    mapx_req_addr <= map_addr(map_x_base_addr, map_index);
                    mapx_req_valid <= 1'b1;
                    map_state <= ST_WAIT_X;
                end
            end
            ST_WAIT_X: begin
                if (mapx_req_valid && mapx_req_ready) begin
                    mapx_req_valid <= 1'b0;
                    mapx_rsp_ready <= 1'b1;
                end
                if (mapx_rsp_valid && mapx_rsp_ready) begin
                    mapx_rsp_ready <= 1'b0;
                    map_x_value <= mapx_rsp_data;
                    mapy_req_addr <= map_addr(map_y_base_addr, map_index);
                    mapy_req_valid <= 1'b1;
                    map_state <= ST_WAIT_Y;
                end
            end
            ST_WAIT_Y: begin
                if (mapy_req_valid && mapy_req_ready) begin
                    mapy_req_valid <= 1'b0;
                    mapy_rsp_ready <= 1'b1;
                end
                if (mapy_rsp_valid && mapy_rsp_ready) begin
                    mapy_rsp_ready <= 1'b0;
                    map_y_value <= mapy_rsp_data;
                    map_valid <= 1'b1;
                end
            end
            default: map_state <= ST_IDLE;
        endcase

        if (map_valid && map_ready) begin
            map_valid <= 1'b0;
            map_index <= map_index + 32'd1;
            map_state <= ST_FETCH_X;
        end

        if (dst_valid && dst_ready) begin
            wr_req_addr <= pixel_addr(dest_base_addr, dest_stride_bytes, dst_x, dst_y);
            wr_req_data <= dst_pixel;
            wr_req_strb <= {(PIXEL_WIDTH/8){1'b1}};
            wr_req_valid <= 1'b1;
            write_inflight <= 1'b1;
        end
        if (wr_req_valid && wr_req_ready) begin
            wr_req_valid <= 1'b0;
            wr_done_ready <= 1'b1;
        end
        if (wr_done_valid && wr_done_ready) begin
            wr_done_ready <= 1'b0;
            write_inflight <= 1'b0;
        end

        if (core_done) begin
            core_done_pending <= 1'b1;
        end

        if ((core_done || core_done_pending) && !write_inflight && !wr_req_valid) begin
            busy <= 1'b0;
            done <= 1'b1;
            map_state <= ST_IDLE;
            map_valid <= 1'b0;
            core_done_pending <= 1'b0;
        end
    end
end

endmodule
