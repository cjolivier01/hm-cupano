module pano_axi_single_read #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
) (
    input wire clk,
    input wire rstn,
    input wire req_valid,
    output wire req_ready,
    input wire [ADDR_WIDTH-1:0] req_addr,
    output reg rsp_valid,
    input wire rsp_ready,
    output reg [DATA_WIDTH-1:0] rsp_data,
    output reg [ADDR_WIDTH-1:0] m_axi_araddr,
    output reg m_axi_arvalid,
    input wire m_axi_arready,
    output wire [7:0] m_axi_arlen,
    output wire [2:0] m_axi_arsize,
    output wire [1:0] m_axi_arburst,
    input wire [DATA_WIDTH-1:0] m_axi_rdata,
    input wire m_axi_rvalid,
    output reg m_axi_rready,
    input wire m_axi_rlast,
    input wire [1:0] m_axi_rresp
);

localparam ST_IDLE = 2'd0;
localparam ST_WAIT_R = 2'd1;
localparam ST_HOLD = 2'd2;

localparam [2:0] AXI_SIZE =
    (DATA_WIDTH == 8) ? 3'd0 :
    (DATA_WIDTH == 16) ? 3'd1 :
    (DATA_WIDTH == 32) ? 3'd2 : 3'd3;

reg [1:0] state;

assign req_ready = (state == ST_IDLE) && !m_axi_arvalid;
assign m_axi_arlen = 8'd0;
assign m_axi_arsize = AXI_SIZE;
assign m_axi_arburst = 2'b01;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        state <= ST_IDLE;
        rsp_valid <= 1'b0;
        rsp_data <= {DATA_WIDTH{1'b0}};
        m_axi_araddr <= {ADDR_WIDTH{1'b0}};
        m_axi_arvalid <= 1'b0;
        m_axi_rready <= 1'b0;
    end else begin
        case (state)
            ST_IDLE: begin
                rsp_valid <= 1'b0;
                m_axi_rready <= 1'b0;
                if (req_valid) begin
                    m_axi_araddr <= req_addr;
                    m_axi_arvalid <= 1'b1;
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        state <= ST_WAIT_R;
                    end else begin
                        state <= ST_WAIT_R;
                    end
                end
            end
            ST_WAIT_R: begin
                if (m_axi_arvalid && m_axi_arready) begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b1;
                end
                if (m_axi_rvalid) begin
                    m_axi_rready <= 1'b0;
                    rsp_data <= m_axi_rdata;
                    rsp_valid <= 1'b1;
                    if (rsp_ready) begin
                        rsp_valid <= 1'b0;
                        state <= ST_IDLE;
                    end else begin
                        state <= ST_HOLD;
                    end
                end
            end
            ST_HOLD: begin
                if (rsp_ready) begin
                    rsp_valid <= 1'b0;
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
