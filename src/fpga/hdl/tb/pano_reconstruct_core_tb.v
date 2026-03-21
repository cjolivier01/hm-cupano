`timescale 1ns/1ps
module pano_reconstruct_core_tb;
  reg x_odd;
  reg y_odd;
  reg [4*18-1:0] lower00;
  reg [4*18-1:0] lower10;
  reg [4*18-1:0] lower01;
  reg [4*18-1:0] lower11;
  reg [4*18-1:0] lap_pixel;
  wire [4*18-1:0] recon_pixel;

  pano_reconstruct_core uut (
      .x_odd(x_odd),
      .y_odd(y_odd),
      .lower00(lower00),
      .lower10(lower10),
      .lower01(lower01),
      .lower11(lower11),
      .lap_pixel(lap_pixel),
      .recon_pixel(recon_pixel)
  );

  initial begin
    x_odd = 1'b1;
    y_odd = 1'b1;
    lower00 = {18'd255, 18'd160, 18'd120, 18'd80};
    lower10 = {18'd255, 18'd160, 18'd120, 18'd80};
    lower01 = {18'd255, 18'd160, 18'd120, 18'd80};
    lower11 = {18'd255, 18'd160, 18'd120, 18'd80};
    lap_pixel = {18'd255, 18'd16, 18'd8, 18'd4};
    #1;
    if (recon_pixel[17:0] != 18'd84) $fatal(1, "blue channel mismatch: %0d", recon_pixel[17:0]);
    if (recon_pixel[35:18] != 18'd128) $fatal(1, "green channel mismatch: %0d", recon_pixel[35:18]);
    if (recon_pixel[53:36] != 18'd176) $fatal(1, "red channel mismatch: %0d", recon_pixel[53:36]);
    if (recon_pixel[71:54] != 18'd255) $fatal(1, "alpha channel mismatch: %0d", recon_pixel[71:54]);
    $display("pano_reconstruct_core_tb PASS");
    $finish;
  end
endmodule
