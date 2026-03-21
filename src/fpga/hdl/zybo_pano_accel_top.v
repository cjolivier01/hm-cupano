module zybo_pano_accel_top #(
    parameter ADDR_WIDTH = 32,
    parameter PIXEL_WIDTH = 32,
    parameter MAP_WIDTH = 16
) (
    input wire clk,
    input wire rstn,
    input wire cfg_write_valid,
    input wire [7:0] cfg_write_addr,
    input wire [31:0] cfg_write_data,
    input wire cfg_read_valid,
    input wire [7:0] cfg_read_addr,
    output wire [31:0] cfg_read_data,
    input wire downsample_done_ext,
    input wire laplacian_done_ext,
    input wire blend_done_ext,
    input wire reconstruct_done_ext,
    output wire irq,
    output wire busy,
    output wire done,
    output wire error,
    output wire [ADDR_WIDTH-1:0] m_axi_hp0_araddr,
    output wire m_axi_hp0_arvalid,
    input wire m_axi_hp0_arready,
    output wire [7:0] m_axi_hp0_arlen,
    output wire [2:0] m_axi_hp0_arsize,
    output wire [1:0] m_axi_hp0_arburst,
    input wire [PIXEL_WIDTH-1:0] m_axi_hp0_rdata,
    input wire m_axi_hp0_rvalid,
    output wire m_axi_hp0_rready,
    input wire m_axi_hp0_rlast,
    input wire [1:0] m_axi_hp0_rresp,
    output wire [ADDR_WIDTH-1:0] m_axi_hp1_araddr,
    output wire m_axi_hp1_arvalid,
    input wire m_axi_hp1_arready,
    output wire [7:0] m_axi_hp1_arlen,
    output wire [2:0] m_axi_hp1_arsize,
    output wire [1:0] m_axi_hp1_arburst,
    input wire [MAP_WIDTH-1:0] m_axi_hp1_rdata,
    input wire m_axi_hp1_rvalid,
    output wire m_axi_hp1_rready,
    input wire m_axi_hp1_rlast,
    input wire [1:0] m_axi_hp1_rresp,
    output wire [ADDR_WIDTH-1:0] m_axi_hp2_araddr,
    output wire m_axi_hp2_arvalid,
    input wire m_axi_hp2_arready,
    output wire [7:0] m_axi_hp2_arlen,
    output wire [2:0] m_axi_hp2_arsize,
    output wire [1:0] m_axi_hp2_arburst,
    input wire [MAP_WIDTH-1:0] m_axi_hp2_rdata,
    input wire m_axi_hp2_rvalid,
    output wire m_axi_hp2_rready,
    input wire m_axi_hp2_rlast,
    input wire [1:0] m_axi_hp2_rresp,
    output wire [ADDR_WIDTH-1:0] m_axi_hp3_awaddr,
    output wire m_axi_hp3_awvalid,
    input wire m_axi_hp3_awready,
    output wire [PIXEL_WIDTH-1:0] m_axi_hp3_wdata,
    output wire [(PIXEL_WIDTH/8)-1:0] m_axi_hp3_wstrb,
    output wire m_axi_hp3_wvalid,
    input wire m_axi_hp3_wready,
    input wire [1:0] m_axi_hp3_bresp,
    input wire m_axi_hp3_bvalid,
    output wire m_axi_hp3_bready
);

wire remap_start;
wire copy_start;
wire downsample_start;
wire laplacian_start;
wire blend_start;
wire reconstruct_start;
wire remap_done;
wire copy_done;
wire [159:0] src0_desc;
wire [159:0] src1_desc;
wire [159:0] src2_desc;
wire [159:0] dest_desc;
wire [287:0] remap_desc;
wire [95:0] copy_desc;
wire [127:0] blend_desc;
wire [95:0] pyramid_desc;

wire remap_busy;
wire copy_busy;

wire remap_src_req_valid;
wire remap_src_req_ready;
wire [ADDR_WIDTH-1:0] remap_src_req_addr;
wire remap_src_rsp_valid;
wire remap_src_rsp_ready;
wire [PIXEL_WIDTH-1:0] remap_src_rsp_data;
wire copy_src_req_valid;
wire copy_src_req_ready;
wire [ADDR_WIDTH-1:0] copy_src_req_addr;
wire copy_src_rsp_valid;
wire copy_src_rsp_ready;
wire [PIXEL_WIDTH-1:0] copy_src_rsp_data;

wire mapx_rd_req_valid;
wire mapx_rd_req_ready;
wire [ADDR_WIDTH-1:0] mapx_rd_req_addr;
wire mapx_rd_rsp_valid;
wire mapx_rd_rsp_ready;
wire [MAP_WIDTH-1:0] mapx_rd_rsp_data;
wire mapy_rd_req_valid;
wire mapy_rd_req_ready;
wire [ADDR_WIDTH-1:0] mapy_rd_req_addr;
wire mapy_rd_rsp_valid;
wire mapy_rd_rsp_ready;
wire [MAP_WIDTH-1:0] mapy_rd_rsp_data;

wire remap_wr_req_valid;
wire [ADDR_WIDTH-1:0] remap_wr_req_addr;
wire [PIXEL_WIDTH-1:0] remap_wr_req_data;
wire [(PIXEL_WIDTH/8)-1:0] remap_wr_req_strb;
wire remap_wr_done_ready;
wire copy_wr_req_valid;
wire [ADDR_WIDTH-1:0] copy_wr_req_addr;
wire [PIXEL_WIDTH-1:0] copy_wr_req_data;
wire [(PIXEL_WIDTH/8)-1:0] copy_wr_req_strb;
wire copy_wr_done_ready;

wire src_rd_req_valid_mux;
wire [ADDR_WIDTH-1:0] src_rd_req_addr_mux;
wire src_rd_req_ready_mux;
wire src_rd_rsp_valid_mux;
wire src_rd_rsp_ready_mux;
wire [PIXEL_WIDTH-1:0] src_rd_rsp_data_mux;
wire wr_req_valid_mux;
wire [ADDR_WIDTH-1:0] wr_req_addr_mux;
wire [PIXEL_WIDTH-1:0] wr_req_data_mux;
wire [(PIXEL_WIDTH/8)-1:0] wr_req_strb_mux;
wire wr_req_ready_mux;
wire wr_done_valid_mux;
wire wr_done_ready_mux;
wire m_axi_hp3_req_ready;
wire m_axi_hp3_done_valid;

assign src_rd_req_valid_mux = remap_busy ? remap_src_req_valid : copy_src_req_valid;
assign src_rd_req_addr_mux = remap_busy ? remap_src_req_addr : copy_src_req_addr;
assign src_rd_rsp_ready_mux = remap_busy ? remap_src_rsp_ready : copy_src_rsp_ready;
assign wr_req_valid_mux = remap_busy ? remap_wr_req_valid : copy_wr_req_valid;
assign wr_req_addr_mux = remap_busy ? remap_wr_req_addr : copy_wr_req_addr;
assign wr_req_data_mux = remap_busy ? remap_wr_req_data : copy_wr_req_data;
assign wr_req_strb_mux = remap_busy ? remap_wr_req_strb : copy_wr_req_strb;
assign wr_done_ready_mux = remap_busy ? remap_wr_done_ready : copy_wr_done_ready;
assign remap_src_req_ready = remap_busy ? src_rd_req_ready_mux : 1'b0;
assign copy_src_req_ready = copy_busy ? src_rd_req_ready_mux : 1'b0;
assign remap_src_rsp_valid = remap_busy ? src_rd_rsp_valid_mux : 1'b0;
assign copy_src_rsp_valid = copy_busy ? src_rd_rsp_valid_mux : 1'b0;
assign remap_src_rsp_data = src_rd_rsp_data_mux;
assign copy_src_rsp_data = src_rd_rsp_data_mux;
assign wr_req_ready_mux = m_axi_hp3_req_ready;
assign wr_done_valid_mux = m_axi_hp3_done_valid;

pano_accel_top ctrl (
    .clk(clk),
    .rstn(rstn),
    .cfg_write_valid(cfg_write_valid),
    .cfg_write_addr(cfg_write_addr),
    .cfg_write_data(cfg_write_data),
    .cfg_read_valid(cfg_read_valid),
    .cfg_read_addr(cfg_read_addr),
    .cfg_read_data(cfg_read_data),
    .remap_done(remap_done),
    .copy_done(copy_done),
    .downsample_done(downsample_done_ext),
    .laplacian_done(laplacian_done_ext),
    .blend_done(blend_done_ext),
    .reconstruct_done(reconstruct_done_ext),
    .remap_start(remap_start),
    .copy_start(copy_start),
    .downsample_start(downsample_start),
    .laplacian_start(laplacian_start),
    .blend_start(blend_start),
    .reconstruct_start(reconstruct_start),
    .irq(irq),
    .busy(busy),
    .done(done),
    .error(error),
    .src0_desc(src0_desc),
    .src1_desc(src1_desc),
    .src2_desc(src2_desc),
    .dest_desc(dest_desc),
    .remap_desc(remap_desc),
    .copy_desc(copy_desc),
    .blend_desc(blend_desc),
    .pyramid_desc(pyramid_desc)
);

zybo_pano_remap_engine #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .PIXEL_WIDTH(PIXEL_WIDTH),
    .MAP_WIDTH(MAP_WIDTH)
) remap_engine (
    .clk(clk),
    .rstn(rstn),
    .start(remap_start),
    .src_base_addr(src0_desc[31:0]),
    .dest_base_addr(dest_desc[31:0]),
    .map_x_base_addr(remap_desc[31:0]),
    .map_y_base_addr(remap_desc[95:64]),
    .src_stride_bytes(src0_desc[95:64]),
    .dest_stride_bytes(dest_desc[95:64]),
    .src_width(src0_desc[111:96]),
    .src_height(src0_desc[127:112]),
    .remap_width(remap_desc[143:128]),
    .remap_height(remap_desc[159:144]),
    .offset_x(remap_desc[175:160]),
    .offset_y(remap_desc[191:176]),
    .no_unmapped_write(remap_desc[192]),
    .default_pixel({PIXEL_WIDTH{1'b0}}),
    .mapx_req_valid(mapx_rd_req_valid),
    .mapx_req_ready(mapx_rd_req_ready),
    .mapx_req_addr(mapx_rd_req_addr),
    .mapx_rsp_valid(mapx_rd_rsp_valid),
    .mapx_rsp_ready(mapx_rd_rsp_ready),
    .mapx_rsp_data(mapx_rd_rsp_data),
    .mapy_req_valid(mapy_rd_req_valid),
    .mapy_req_ready(mapy_rd_req_ready),
    .mapy_req_addr(mapy_rd_req_addr),
    .mapy_rsp_valid(mapy_rd_rsp_valid),
    .mapy_rsp_ready(mapy_rd_rsp_ready),
    .mapy_rsp_data(mapy_rd_rsp_data),
    .src_req_valid(remap_src_req_valid),
    .src_req_ready(remap_src_req_ready),
    .src_req_addr(remap_src_req_addr),
    .src_rsp_valid(remap_src_rsp_valid),
    .src_rsp_ready(remap_src_rsp_ready),
    .src_rsp_data(remap_src_rsp_data),
    .wr_req_valid(remap_wr_req_valid),
    .wr_req_ready(wr_req_ready_mux),
    .wr_req_addr(remap_wr_req_addr),
    .wr_req_data(remap_wr_req_data),
    .wr_req_strb(remap_wr_req_strb),
    .wr_done_valid(wr_done_valid_mux),
    .wr_done_ready(remap_wr_done_ready),
    .busy(remap_busy),
    .done(remap_done)
);

pano_copy_roi_engine #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(PIXEL_WIDTH)
) copy_engine (
    .clk(clk),
    .rstn(rstn),
    .start(copy_start),
    .src_base_addr(src0_desc[31:0]),
    .dest_base_addr(dest_desc[31:0]),
    .src_stride_bytes(src0_desc[95:64]),
    .dest_stride_bytes(dest_desc[95:64]),
    .src_x(copy_desc[15:0]),
    .src_y(copy_desc[31:16]),
    .dest_x(copy_desc[47:32]),
    .dest_y(copy_desc[63:48]),
    .width(copy_desc[79:64]),
    .height(copy_desc[95:80]),
    .rd_req_valid(copy_src_req_valid),
    .rd_req_ready(copy_src_req_ready),
    .rd_req_addr(copy_src_req_addr),
    .rd_rsp_valid(copy_src_rsp_valid),
    .rd_rsp_ready(copy_src_rsp_ready),
    .rd_rsp_data(copy_src_rsp_data),
    .wr_req_valid(copy_wr_req_valid),
    .wr_req_ready(wr_req_ready_mux),
    .wr_req_addr(copy_wr_req_addr),
    .wr_req_data(copy_wr_req_data),
    .wr_req_strb(copy_wr_req_strb),
    .wr_done_valid(wr_done_valid_mux),
    .wr_done_ready(copy_wr_done_ready),
    .busy(copy_busy),
    .done(copy_done)
);

pano_axi_single_read #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(PIXEL_WIDTH)
) src_reader (
    .clk(clk),
    .rstn(rstn),
    .req_valid(src_rd_req_valid_mux),
    .req_ready(src_rd_req_ready_mux),
    .req_addr(src_rd_req_addr_mux),
    .rsp_valid(src_rd_rsp_valid_mux),
    .rsp_ready(src_rd_rsp_ready_mux),
    .rsp_data(src_rd_rsp_data_mux),
    .m_axi_araddr(m_axi_hp0_araddr),
    .m_axi_arvalid(m_axi_hp0_arvalid),
    .m_axi_arready(m_axi_hp0_arready),
    .m_axi_arlen(m_axi_hp0_arlen),
    .m_axi_arsize(m_axi_hp0_arsize),
    .m_axi_arburst(m_axi_hp0_arburst),
    .m_axi_rdata(m_axi_hp0_rdata),
    .m_axi_rvalid(m_axi_hp0_rvalid),
    .m_axi_rready(m_axi_hp0_rready),
    .m_axi_rlast(m_axi_hp0_rlast),
    .m_axi_rresp(m_axi_hp0_rresp)
);

pano_axi_single_read #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(MAP_WIDTH)
) mapx_reader (
    .clk(clk),
    .rstn(rstn),
    .req_valid(mapx_rd_req_valid),
    .req_ready(mapx_rd_req_ready),
    .req_addr(mapx_rd_req_addr),
    .rsp_valid(mapx_rd_rsp_valid),
    .rsp_ready(mapx_rd_rsp_ready),
    .rsp_data(mapx_rd_rsp_data),
    .m_axi_araddr(m_axi_hp1_araddr),
    .m_axi_arvalid(m_axi_hp1_arvalid),
    .m_axi_arready(m_axi_hp1_arready),
    .m_axi_arlen(m_axi_hp1_arlen),
    .m_axi_arsize(m_axi_hp1_arsize),
    .m_axi_arburst(m_axi_hp1_arburst),
    .m_axi_rdata(m_axi_hp1_rdata),
    .m_axi_rvalid(m_axi_hp1_rvalid),
    .m_axi_rready(m_axi_hp1_rready),
    .m_axi_rlast(m_axi_hp1_rlast),
    .m_axi_rresp(m_axi_hp1_rresp)
);

pano_axi_single_read #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(MAP_WIDTH)
) mapy_reader (
    .clk(clk),
    .rstn(rstn),
    .req_valid(mapy_rd_req_valid),
    .req_ready(mapy_rd_req_ready),
    .req_addr(mapy_rd_req_addr),
    .rsp_valid(mapy_rd_rsp_valid),
    .rsp_ready(mapy_rd_rsp_ready),
    .rsp_data(mapy_rd_rsp_data),
    .m_axi_araddr(m_axi_hp2_araddr),
    .m_axi_arvalid(m_axi_hp2_arvalid),
    .m_axi_arready(m_axi_hp2_arready),
    .m_axi_arlen(m_axi_hp2_arlen),
    .m_axi_arsize(m_axi_hp2_arsize),
    .m_axi_arburst(m_axi_hp2_arburst),
    .m_axi_rdata(m_axi_hp2_rdata),
    .m_axi_rvalid(m_axi_hp2_rvalid),
    .m_axi_rready(m_axi_hp2_rready),
    .m_axi_rlast(m_axi_hp2_rlast),
    .m_axi_rresp(m_axi_hp2_rresp)
);

pano_axi_single_write #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .DATA_WIDTH(PIXEL_WIDTH)
) writer (
    .clk(clk),
    .rstn(rstn),
    .req_valid(wr_req_valid_mux),
    .req_ready(m_axi_hp3_req_ready),
    .req_addr(wr_req_addr_mux),
    .req_data(wr_req_data_mux),
    .req_strb(wr_req_strb_mux),
    .done_valid(m_axi_hp3_done_valid),
    .done_ready(wr_done_ready_mux),
    .m_axi_awaddr(m_axi_hp3_awaddr),
    .m_axi_awvalid(m_axi_hp3_awvalid),
    .m_axi_awready(m_axi_hp3_awready),
    .m_axi_wdata(m_axi_hp3_wdata),
    .m_axi_wstrb(m_axi_hp3_wstrb),
    .m_axi_wvalid(m_axi_hp3_wvalid),
    .m_axi_wready(m_axi_hp3_wready),
    .m_axi_bresp(m_axi_hp3_bresp),
    .m_axi_bvalid(m_axi_hp3_bvalid),
    .m_axi_bready(m_axi_hp3_bready)
);

endmodule
