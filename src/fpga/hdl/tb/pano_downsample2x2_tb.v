`timescale 1ns/1ps
module pano_downsample2x2_tb;
  reg [4*18-1:0] image1_p00;
  reg [4*18-1:0] image1_p01;
  reg [4*18-1:0] image1_p10;
  reg [4*18-1:0] image1_p11;
  reg [4*18-1:0] image2_p00;
  reg [4*18-1:0] image2_p01;
  reg [4*18-1:0] image2_p10;
  reg [4*18-1:0] image2_p11;
  reg [15:0] mask_p00;
  reg [15:0] mask_p01;
  reg [15:0] mask_p10;
  reg [15:0] mask_p11;
  wire [4*18-1:0] image1_out;
  wire [4*18-1:0] image2_out;
  wire [15:0] mask_out;

  pano_downsample2x2 uut (
      .image1_p00(image1_p00), .image1_p01(image1_p01), .image1_p10(image1_p10), .image1_p11(image1_p11),
      .image2_p00(image2_p00), .image2_p01(image2_p01), .image2_p10(image2_p10), .image2_p11(image2_p11),
      .mask_p00(mask_p00), .mask_p01(mask_p01), .mask_p10(mask_p10), .mask_p11(mask_p11),
      .image1_out(image1_out), .image2_out(image2_out), .mask_out(mask_out)
  );

  initial begin
    image1_p00 = {18'd255, 18'd40, 18'd30, 18'd20};
    image1_p01 = {18'd255, 18'd80, 18'd50, 18'd40};
    image1_p10 = {18'd255, 18'd120, 18'd70, 18'd60};
    image1_p11 = {18'd255, 18'd160, 18'd90, 18'd80};
    image2_p00 = {18'd255, 18'd20, 18'd10, 18'd0};
    image2_p01 = {18'd255, 18'd40, 18'd30, 18'd20};
    image2_p10 = {18'd255, 18'd60, 18'd50, 18'd40};
    image2_p11 = {18'd255, 18'd80, 18'd70, 18'd60};
    mask_p00 = 16'd65535;
    mask_p01 = 16'd65535;
    mask_p10 = 16'd0;
    mask_p11 = 16'd0;
    #1;
    if (image1_out[17:0] != 18'd50) $fatal(1, "image1 red average mismatch: %0d", image1_out[17:0]);
    if (image2_out[17:0] != 18'd30) $fatal(1, "image2 red average mismatch: %0d", image2_out[17:0]);
    if (mask_out != 16'd32767) $fatal(1, "mask average mismatch: %0d", mask_out);
    $display("pano_downsample2x2_tb PASS");
    $finish;
  end
endmodule
