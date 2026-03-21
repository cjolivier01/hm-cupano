`timescale 1ns/1ps
module pano_two_image_assets_tb #(
    parameter ADDR_WIDTH = 32,
    parameter INPUT_W = 16,
    parameter INPUT_H = 8,
    parameter OVERLAP = 4,
    parameter PAD = 2,
    parameter TIMEOUT_CYCLES = 1000 + (INPUT_W * INPUT_H * 2048)
);
  localparam X2 = INPUT_W - OVERLAP;
  localparam CANVAS_W = INPUT_W + X2;
  localparam CANVAS_H = INPUT_H;
  localparam BLEND_W = OVERLAP + (2 * PAD);
  localparam BLEND_H = INPUT_H;
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
  localparam MASK_HIGH_BASE = 32'h000A_0000;

  reg clk;
  reg rstn;
  reg start;
  reg host_wr_en;
  reg [ADDR_WIDTH-1:0] host_wr_addr;
  reg [31:0] host_wr_data;
  reg host_rd_en;
  reg [ADDR_WIDTH-1:0] host_rd_addr;
  wire busy;
  wire done;
  wire [31:0] host_rd_data;
  wire host_rd_valid;

  reg [31:0] left_words [0:(INPUT_W * INPUT_H) - 1];
  reg [31:0] right_words [0:(INPUT_W * INPUT_H) - 1];
  reg [15:0] map1_x_words [0:(INPUT_W * INPUT_H) - 1];
  reg [15:0] map1_y_words [0:(INPUT_W * INPUT_H) - 1];
  reg [15:0] map2_x_words [0:(INPUT_W * INPUT_H) - 1];
  reg [15:0] map2_y_words [0:(INPUT_W * INPUT_H) - 1];
  reg [15:0] mask_high_words [0:(BLEND_W * BLEND_H) - 1];
  reg [31:0] canvas_words [0:(CANVAS_W * CANVAS_H) - 1];

  integer idx;
  reg [31:0] read_word_data;

  pano_two_image_assets_pipeline #(
      .ADDR_WIDTH(ADDR_WIDTH),
      .INPUT_W(INPUT_W),
      .INPUT_H(INPUT_H),
      .OVERLAP(OVERLAP),
      .PAD(PAD),
      .INIT_MEMS(0)
  ) uut (
      .clk(clk),
      .rstn(rstn),
      .start(start),
      .host_wr_en(host_wr_en),
      .host_wr_addr(host_wr_addr),
      .host_wr_data(host_wr_data),
      .host_rd_en(host_rd_en),
      .host_rd_addr(host_rd_addr),
      .busy(busy),
      .done(done),
      .host_rd_data(host_rd_data),
      .host_rd_valid(host_rd_valid)
  );

  always #5 clk = ~clk;

  task write_word32;
    input [ADDR_WIDTH-1:0] addr;
    input [31:0] data;
    begin
      @(negedge clk);
      host_wr_en = 1'b1;
      host_wr_addr = addr;
      host_wr_data = data;
      @(negedge clk);
      host_wr_en = 1'b0;
      host_wr_addr = {ADDR_WIDTH{1'b0}};
      host_wr_data = 32'd0;
    end
  endtask

  task write_half16;
    input [ADDR_WIDTH-1:0] addr;
    input [15:0] data;
    begin
      write_word32(addr, {16'd0, data});
    end
  endtask

  task read_word32;
    input [ADDR_WIDTH-1:0] addr;
    output [31:0] data;
    begin
      @(negedge clk);
      host_rd_en = 1'b1;
      host_rd_addr = addr;
      @(negedge clk);
      host_rd_en = 1'b0;
      host_rd_addr = {ADDR_WIDTH{1'b0}};
      if (!host_rd_valid) begin
        $fatal(1, "host read did not complete for addr 0x%08x", addr);
      end
      data = host_rd_data;
    end
  endtask

  initial begin
    if ($test$plusargs("dump_vcd")) begin
      $dumpfile("pano_two_image_assets.vcd");
      $dumpvars(0, pano_two_image_assets_tb);
    end
  end

  initial begin
    clk = 1'b0;
    rstn = 1'b0;
    start = 1'b0;
    host_wr_en = 1'b0;
    host_wr_addr = {ADDR_WIDTH{1'b0}};
    host_wr_data = 32'd0;
    host_rd_en = 1'b0;
    host_rd_addr = {ADDR_WIDTH{1'b0}};
    read_word_data = 32'd0;

    $readmemh("left.hex", left_words);
    $readmemh("right.hex", right_words);
    $readmemh("map1_x.hex", map1_x_words);
    $readmemh("map1_y.hex", map1_y_words);
    $readmemh("map2_x.hex", map2_x_words);
    $readmemh("map2_y.hex", map2_y_words);
    $readmemh("mask_high.hex", mask_high_words);

    repeat (4) @(posedge clk);
    rstn = 1'b1;

    for (idx = 0; idx < CANVAS_W * CANVAS_H; idx = idx + 1) begin
      write_word32(CANVAS_BASE + (idx * 4), 32'd0);
    end
    for (idx = 0; idx < BLEND_W * BLEND_H; idx = idx + 1) begin
      write_word32(BLEND_LEFT_BASE + (idx * 4), 32'd0);
      write_word32(BLEND_RIGHT_BASE + (idx * 4), 32'd0);
      write_word32(BLEND_OUT_BASE + (idx * 4), 32'd0);
    end

    for (idx = 0; idx < INPUT_W * INPUT_H; idx = idx + 1) begin
      write_word32(LEFT_BASE + (idx * 4), left_words[idx]);
      write_word32(RIGHT_BASE + (idx * 4), right_words[idx]);
      write_half16(MAP1_X_BASE + (idx * 2), map1_x_words[idx]);
      write_half16(MAP1_Y_BASE + (idx * 2), map1_y_words[idx]);
      write_half16(MAP2_X_BASE + (idx * 2), map2_x_words[idx]);
      write_half16(MAP2_Y_BASE + (idx * 2), map2_y_words[idx]);
    end
    for (idx = 0; idx < BLEND_W * BLEND_H; idx = idx + 1) begin
      write_half16(MASK_HIGH_BASE + (idx * 2), mask_high_words[idx]);
    end

    @(negedge clk);
    start = 1'b1;
    @(negedge clk);
    start = 1'b0;

    wait (done);
    for (idx = 0; idx < CANVAS_W * CANVAS_H; idx = idx + 1) begin
      read_word32(CANVAS_BASE + (idx * 4), read_word_data);
      canvas_words[idx] = read_word_data;
    end

    $writememh("canvas_out.hex", canvas_words);
    $display("pano_two_image_assets_tb PASS");
    $finish;
  end

  initial begin
    repeat (TIMEOUT_CYCLES) @(posedge clk);
    $fatal(1, "pano_two_image_assets_tb timeout");
  end
endmodule
