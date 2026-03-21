module pano_copy_roi_engine #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter PIXEL_BYTES = 4
) (
    input wire clk,
    input wire rstn,
    input wire start,
    input wire [ADDR_WIDTH-1:0] src_base_addr,
    input wire [ADDR_WIDTH-1:0] dest_base_addr,
    input wire [31:0] src_stride_bytes,
    input wire [31:0] dest_stride_bytes,
    input wire [15:0] src_x,
    input wire [15:0] src_y,
    input wire [15:0] dest_x,
    input wire [15:0] dest_y,
    input wire [15:0] width,
    input wire [15:0] height,
    output reg rd_req_valid,
    input wire rd_req_ready,
    output reg [ADDR_WIDTH-1:0] rd_req_addr,
    input wire rd_rsp_valid,
    output reg rd_rsp_ready,
    input wire [DATA_WIDTH-1:0] rd_rsp_data,
    output reg wr_req_valid,
    input wire wr_req_ready,
    output reg [ADDR_WIDTH-1:0] wr_req_addr,
    output reg [DATA_WIDTH-1:0] wr_req_data,
    output reg [(DATA_WIDTH/8)-1:0] wr_req_strb,
    input wire wr_done_valid,
    output reg wr_done_ready,
    output reg busy,
    output reg done
);

localparam ST_IDLE = 3'd0;
localparam ST_ISSUE_READ = 3'd1;
localparam ST_WAIT_READ = 3'd2;
localparam ST_ISSUE_WRITE = 3'd3;
localparam ST_WAIT_WRITE = 3'd4;

reg [2:0] state;
reg [15:0] x_pos;
reg [15:0] y_pos;

function [ADDR_WIDTH-1:0] pixel_addr;
    input [ADDR_WIDTH-1:0] base_addr;
    input [31:0] stride_bytes;
    input [15:0] x;
    input [15:0] y;
    begin
        pixel_addr = base_addr + (y * stride_bytes) + (x * PIXEL_BYTES);
    end
endfunction

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= ST_IDLE;
        x_pos <= 16'd0;
        y_pos <= 16'd0;
        rd_req_valid <= 1'b0;
        rd_req_addr <= {ADDR_WIDTH{1'b0}};
        rd_rsp_ready <= 1'b0;
        wr_req_valid <= 1'b0;
        wr_req_addr <= {ADDR_WIDTH{1'b0}};
        wr_req_data <= {DATA_WIDTH{1'b0}};
        wr_req_strb <= {(DATA_WIDTH/8){1'b1}};
        wr_done_ready <= 1'b0;
        busy <= 1'b0;
        done <= 1'b0;
    end else begin
        done <= 1'b0;
        case (state)
            ST_IDLE: begin
                rd_req_valid <= 1'b0;
                rd_rsp_ready <= 1'b0;
                wr_req_valid <= 1'b0;
                wr_done_ready <= 1'b0;
                busy <= 1'b0;
                if (start) begin
                    x_pos <= 16'd0;
                    y_pos <= 16'd0;
                    busy <= 1'b1;
                    state <= ST_ISSUE_READ;
                end
            end
            ST_ISSUE_READ: begin
                rd_req_addr <= pixel_addr(src_base_addr, src_stride_bytes, src_x + x_pos, src_y + y_pos);
                rd_req_valid <= 1'b1;
                if (rd_req_ready) begin
                    rd_rsp_ready <= 1'b1;
                    state <= ST_WAIT_READ;
                end
            end
            ST_WAIT_READ: begin
                rd_req_valid <= 1'b0;
                if (rd_rsp_valid) begin
                    rd_rsp_ready <= 1'b0;
                    wr_req_addr <= pixel_addr(dest_base_addr, dest_stride_bytes, dest_x + x_pos, dest_y + y_pos);
                    wr_req_data <= rd_rsp_data;
                    wr_req_strb <= {(DATA_WIDTH/8){1'b1}};
                    state <= ST_ISSUE_WRITE;
                end
            end
            ST_ISSUE_WRITE: begin
                wr_req_valid <= 1'b1;
                if (wr_req_ready) begin
                    wr_done_ready <= 1'b1;
                    state <= ST_WAIT_WRITE;
                end
            end
            ST_WAIT_WRITE: begin
                wr_req_valid <= 1'b0;
                if (wr_done_valid) begin
                    wr_done_ready <= 1'b0;
                    if (x_pos == width - 1) begin
                        x_pos <= 16'd0;
                        if (y_pos == height - 1) begin
                            y_pos <= 16'd0;
                            busy <= 1'b0;
                            done <= 1'b1;
                            state <= ST_IDLE;
                        end else begin
                            y_pos <= y_pos + 16'd1;
                            state <= ST_ISSUE_READ;
                        end
                    end else begin
                        x_pos <= x_pos + 16'd1;
                        state <= ST_ISSUE_READ;
                    end
                end
            end
            default: begin
                state <= ST_IDLE;
            end
        endcase
    end
end

endmodule
