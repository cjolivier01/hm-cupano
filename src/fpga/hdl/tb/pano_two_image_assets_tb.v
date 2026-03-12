`timescale 1ns/1ps
module pano_two_image_assets_tb;
  reg clk;
  reg rstn;
  reg start;
  wire busy;
  wire done;

  pano_two_image_assets_pipeline uut (
      .clk(clk),
      .rstn(rstn),
      .start(start),
      .busy(busy),
      .done(done)
  );

  always #5 clk = ~clk;

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

    $readmemh("left.hex", uut.left_in_mem);
    $readmemh("right.hex", uut.right_in_mem);

    repeat (4) @(posedge clk);
    rstn = 1'b1;
    @(posedge clk);
    start = 1'b1;
    @(posedge clk);
    start = 1'b0;

    wait (done);
    @(posedge clk);
    $writememh("canvas_out.hex", uut.canvas_mem);
    $display("pano_two_image_assets_tb PASS");
    $finish;
  end

  initial begin
    #2000000;
    $fatal(1, "pano_two_image_assets_tb timeout");
  end
endmodule
