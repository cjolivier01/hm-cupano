module pano_remap_core #(
    parameter DATA_WIDTH = 32,
    parameter COORD_WIDTH = 16,
    parameter ADDR_WIDTH = 32,
    parameter STRIDE_WIDTH = 16,
    parameter UNMAPPED_VALUE = 16'hffff
) (
    input wire clk,
    input wire rstn,
    input wire start,
    input wire [15:0] remap_width,
    input wire [15:0] remap_height,
    input wire [15:0] src_width,
    input wire [15:0] src_height,
    input wire [ADDR_WIDTH-1:0] src_base_addr,
    input wire [STRIDE_WIDTH-1:0] src_stride_bytes,
    input wire [STRIDE_WIDTH-1:0] bytes_per_pixel,
    input wire signed [15:0] offset_x,
    input wire signed [15:0] offset_y,
    input wire no_unmapped_write,
    input wire [DATA_WIDTH-1:0] default_pixel,
    input wire map_valid,
    output reg map_ready,
    input wire [COORD_WIDTH-1:0] map_x,
    input wire [COORD_WIDTH-1:0] map_y,
    output reg src_req_valid,
    input wire src_req_ready,
    output reg [ADDR_WIDTH-1:0] src_req_addr,
    input wire src_rsp_valid,
    input wire [DATA_WIDTH-1:0] src_rsp_data,
    output reg dst_valid,
    input wire dst_ready,
    output reg signed [15:0] dst_x,
    output reg signed [15:0] dst_y,
    output reg [DATA_WIDTH-1:0] dst_pixel,
    output reg busy,
    output reg done
);

localparam ST_IDLE = 3'd0;
localparam ST_WAIT_MAP = 3'd1;
localparam ST_ISSUE_REQ = 3'd2;
localparam ST_WAIT_RSP = 3'd3;
localparam ST_EMIT = 3'd4;

reg [2:0] state;
reg [15:0] local_x;
reg [15:0] local_y;
reg [COORD_WIDTH-1:0] map_x_hold;
reg [COORD_WIDTH-1:0] map_y_hold;
reg emit_after_response;
reg skip_emit;

function [ADDR_WIDTH-1:0] pixel_addr;
    input [COORD_WIDTH-1:0] x;
    input [COORD_WIDTH-1:0] y;
    begin
        pixel_addr = src_base_addr + (y * src_stride_bytes) + (x * bytes_per_pixel);
    end
endfunction

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= ST_IDLE;
        local_x <= 16'd0;
        local_y <= 16'd0;
        map_x_hold <= {COORD_WIDTH{1'b0}};
        map_y_hold <= {COORD_WIDTH{1'b0}};
        src_req_valid <= 1'b0;
        src_req_addr <= {ADDR_WIDTH{1'b0}};
        dst_valid <= 1'b0;
        dst_x <= 16'sd0;
        dst_y <= 16'sd0;
        dst_pixel <= {DATA_WIDTH{1'b0}};
        map_ready <= 1'b0;
        busy <= 1'b0;
        done <= 1'b0;
        emit_after_response <= 1'b0;
        skip_emit <= 1'b0;
    end else begin
        done <= 1'b0;
        map_ready <= 1'b0;

        case (state)
            ST_IDLE: begin
                src_req_valid <= 1'b0;
                dst_valid <= 1'b0;
                busy <= 1'b0;
                if (start) begin
                    local_x <= 16'd0;
                    local_y <= 16'd0;
                    busy <= 1'b1;
                    state <= ST_WAIT_MAP;
                end
            end
            ST_WAIT_MAP: begin
                map_ready <= 1'b1;
                if (map_valid) begin
                    map_x_hold <= map_x;
                    map_y_hold <= map_y;
                    dst_x <= offset_x + $signed({1'b0, local_x});
                    dst_y <= offset_y + $signed({1'b0, local_y});
                    if (map_x == UNMAPPED_VALUE && no_unmapped_write) begin
                        skip_emit <= 1'b1;
                        state <= ST_EMIT;
                    end else if (map_x >= src_width || map_y >= src_height) begin
                        skip_emit <= 1'b0;
                        dst_pixel <= default_pixel;
                        state <= ST_EMIT;
                    end else begin
                        skip_emit <= 1'b0;
                        src_req_addr <= pixel_addr(map_x, map_y);
                        src_req_valid <= 1'b1;
                        state <= ST_ISSUE_REQ;
                    end
                end
            end
            ST_ISSUE_REQ: begin
                if (src_req_ready) begin
                    src_req_valid <= 1'b0;
                    state <= ST_WAIT_RSP;
                end
            end
            ST_WAIT_RSP: begin
                if (src_rsp_valid) begin
                    dst_pixel <= src_rsp_data;
                    state <= ST_EMIT;
                end
            end
            ST_EMIT: begin
                if (skip_emit) begin
                    if (local_x == remap_width - 1) begin
                        local_x <= 16'd0;
                        if (local_y == remap_height - 1) begin
                            local_y <= 16'd0;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            local_y <= local_y + 16'd1;
                            state <= ST_WAIT_MAP;
                        end
                    end else begin
                        local_x <= local_x + 16'd1;
                        state <= ST_WAIT_MAP;
                    end
                end else begin
                    dst_valid <= 1'b1;
                    if (dst_valid && dst_ready) begin
                        dst_valid <= 1'b0;
                        if (local_x == remap_width - 1) begin
                            local_x <= 16'd0;
                            if (local_y == remap_height - 1) begin
                                local_y <= 16'd0;
                                busy <= 1'b0;
                                done <= 1'b1;
                                state <= ST_IDLE;
                            end else begin
                                local_y <= local_y + 16'd1;
                                state <= ST_WAIT_MAP;
                            end
                        end else begin
                            local_x <= local_x + 16'd1;
                            state <= ST_WAIT_MAP;
                        end
                    end
                end
            end
            default: state <= ST_IDLE;
        endcase
    end
end

endmodule
