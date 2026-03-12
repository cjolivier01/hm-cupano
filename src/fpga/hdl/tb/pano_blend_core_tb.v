`timescale 1ns/1ps
module pano_blend_core_tb;
  reg [4*18-1:0] lap1_pixel;
  reg [4*18-1:0] lap2_pixel;
  reg [15:0] mask_weight;
  wire [4*18-1:0] blended_pixel;

  pano_blend_core uut (
      .lap1_pixel(lap1_pixel),
      .lap2_pixel(lap2_pixel),
      .mask_weight(mask_weight),
      .blended_pixel(blended_pixel)
  );

  initial begin
    lap1_pixel = {18'd255, 18'd90, 18'd70, 18'd50};
    lap2_pixel = {18'd255, 18'd30, 18'd20, 18'd10};
    mask_weight = 16'd32768;
    #1;
    if (blended_pixel[17:0] < 18'd29 || blended_pixel[17:0] > 18'd31) $fatal(1, "blue channel mismatch");
    if (blended_pixel[35:18] < 18'd44 || blended_pixel[35:18] > 18'd46) $fatal(1, "green channel mismatch");
    if (blended_pixel[53:36] < 18'd59 || blended_pixel[53:36] > 18'd61) $fatal(1, "red channel mismatch");
    if (blended_pixel[71:54] != 18'd255) $fatal(1, "alpha mismatch");
    $display("pano_blend_core_tb PASS");
    $finish;
  end
endmodule
