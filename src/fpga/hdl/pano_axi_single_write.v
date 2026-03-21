module pano_axi_single_write #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter STRB_WIDTH = DATA_WIDTH / 8
) (
    input wire clk,
    input wire rstn,
    input wire req_valid,
    output wire req_ready,
    input wire [ADDR_WIDTH-1:0] req_addr,
    input wire [DATA_WIDTH-1:0] req_data,
    input wire [STRB_WIDTH-1:0] req_strb,
    output reg done_valid,
    input wire done_ready,
    output reg [ADDR_WIDTH-1:0] m_axi_awaddr,
    output reg m_axi_awvalid,
    input wire m_axi_awready,
    output reg [DATA_WIDTH-1:0] m_axi_wdata,
    output reg [STRB_WIDTH-1:0] m_axi_wstrb,
    output reg m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire m_axi_bvalid,
    output reg m_axi_bready
);

localparam ST_IDLE = 2'd0;
localparam ST_WAIT_B = 2'd1;
localparam ST_HOLD = 2'd2;

reg [1:0] state;
reg aw_done;
reg w_done;

assign req_ready = (state == ST_IDLE);

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= ST_IDLE;
        aw_done <= 1'b0;
        w_done <= 1'b0;
        done_valid <= 1'b0;
        m_axi_awaddr <= {ADDR_WIDTH{1'b0}};
        m_axi_awvalid <= 1'b0;
        m_axi_wdata <= {DATA_WIDTH{1'b0}};
        m_axi_wstrb <= {STRB_WIDTH{1'b0}};
        m_axi_wvalid <= 1'b0;
        m_axi_bready <= 1'b0;
    end else begin
        case (state)
            ST_IDLE: begin
                done_valid <= 1'b0;
                m_axi_bready <= 1'b0;
                if (req_valid) begin
                    m_axi_awaddr <= req_addr;
                    m_axi_wdata <= req_data;
                    m_axi_wstrb <= req_strb;
                    m_axi_awvalid <= 1'b1;
                    m_axi_wvalid <= 1'b1;
                    aw_done <= 1'b0;
                    w_done <= 1'b0;
                    state <= ST_WAIT_B;
                end
            end
            ST_WAIT_B: begin
                if (m_axi_awvalid && m_axi_awready) begin
                    m_axi_awvalid <= 1'b0;
                    aw_done <= 1'b1;
                end
                if (m_axi_wvalid && m_axi_wready) begin
                    m_axi_wvalid <= 1'b0;
                    w_done <= 1'b1;
                end
                if ((aw_done || (m_axi_awvalid && m_axi_awready)) && (w_done || (m_axi_wvalid && m_axi_wready))) begin
                    m_axi_bready <= 1'b1;
                end
                if (m_axi_bvalid && m_axi_bready) begin
                    m_axi_bready <= 1'b0;
                    done_valid <= 1'b1;
                    if (done_ready) begin
                        done_valid <= 1'b0;
                        state <= ST_IDLE;
                    end else begin
                        state <= ST_HOLD;
                    end
                end
            end
            ST_HOLD: begin
                if (done_ready) begin
                    done_valid <= 1'b0;
                    state <= ST_IDLE;
                end
            end
            default: begin
                state <= ST_IDLE;
            end
        endcase
    end
end

endmodule
