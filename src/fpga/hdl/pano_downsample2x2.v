module pano_downsample2x2 #(
    parameter CHANNELS = 4,
    parameter DATA_WIDTH = 18,
    parameter MASK_WIDTH = 16
) (
    input wire [CHANNELS*DATA_WIDTH-1:0] image1_p00,
    input wire [CHANNELS*DATA_WIDTH-1:0] image1_p01,
    input wire [CHANNELS*DATA_WIDTH-1:0] image1_p10,
    input wire [CHANNELS*DATA_WIDTH-1:0] image1_p11,
    input wire [CHANNELS*DATA_WIDTH-1:0] image2_p00,
    input wire [CHANNELS*DATA_WIDTH-1:0] image2_p01,
    input wire [CHANNELS*DATA_WIDTH-1:0] image2_p10,
    input wire [CHANNELS*DATA_WIDTH-1:0] image2_p11,
    input wire [MASK_WIDTH-1:0] mask_p00,
    input wire [MASK_WIDTH-1:0] mask_p01,
    input wire [MASK_WIDTH-1:0] mask_p10,
    input wire [MASK_WIDTH-1:0] mask_p11,
    output reg [CHANNELS*DATA_WIDTH-1:0] image1_out,
    output reg [CHANNELS*DATA_WIDTH-1:0] image2_out,
    output reg [MASK_WIDTH-1:0] mask_out
);

integer c;
integer count1;
integer count2;
integer sum1;
integer sum2;
integer alpha1;
integer alpha2;
reg [MASK_WIDTH+1:0] mask_sum;

function signed [DATA_WIDTH-1:0] chan_value;
    input [CHANNELS*DATA_WIDTH-1:0] pixel;
    input integer index;
    begin
        chan_value = pixel[index*DATA_WIDTH +: DATA_WIDTH];
    end
endfunction

always @(*) begin
    image1_out = {CHANNELS*DATA_WIDTH{1'b0}};
    image2_out = {CHANNELS*DATA_WIDTH{1'b0}};
    mask_sum = {2'b0, mask_p00} + {2'b0, mask_p01} + {2'b0, mask_p10} + {2'b0, mask_p11};
    mask_out = mask_sum >> 2;

    for (c = 0; c < CHANNELS; c = c + 1) begin
        if (CHANNELS == 4 && c == 3) begin
            alpha1 = chan_value(image1_p00, 3);
            if (chan_value(image1_p01, 3) > alpha1) alpha1 = chan_value(image1_p01, 3);
            if (chan_value(image1_p10, 3) > alpha1) alpha1 = chan_value(image1_p10, 3);
            if (chan_value(image1_p11, 3) > alpha1) alpha1 = chan_value(image1_p11, 3);
            alpha2 = chan_value(image2_p00, 3);
            if (chan_value(image2_p01, 3) > alpha2) alpha2 = chan_value(image2_p01, 3);
            if (chan_value(image2_p10, 3) > alpha2) alpha2 = chan_value(image2_p10, 3);
            if (chan_value(image2_p11, 3) > alpha2) alpha2 = chan_value(image2_p11, 3);
            image1_out[c*DATA_WIDTH +: DATA_WIDTH] = alpha1[DATA_WIDTH-1:0];
            image2_out[c*DATA_WIDTH +: DATA_WIDTH] = alpha2[DATA_WIDTH-1:0];
        end else begin
            sum1 = 0;
            sum2 = 0;
            count1 = 0;
            count2 = 0;
            if (CHANNELS != 4 || chan_value(image1_p00, 3) != 0) begin sum1 = sum1 + chan_value(image1_p00, c); count1 = count1 + 1; end
            if (CHANNELS != 4 || chan_value(image1_p01, 3) != 0) begin sum1 = sum1 + chan_value(image1_p01, c); count1 = count1 + 1; end
            if (CHANNELS != 4 || chan_value(image1_p10, 3) != 0) begin sum1 = sum1 + chan_value(image1_p10, c); count1 = count1 + 1; end
            if (CHANNELS != 4 || chan_value(image1_p11, 3) != 0) begin sum1 = sum1 + chan_value(image1_p11, c); count1 = count1 + 1; end
            if (CHANNELS != 4 || chan_value(image2_p00, 3) != 0) begin sum2 = sum2 + chan_value(image2_p00, c); count2 = count2 + 1; end
            if (CHANNELS != 4 || chan_value(image2_p01, 3) != 0) begin sum2 = sum2 + chan_value(image2_p01, c); count2 = count2 + 1; end
            if (CHANNELS != 4 || chan_value(image2_p10, 3) != 0) begin sum2 = sum2 + chan_value(image2_p10, c); count2 = count2 + 1; end
            if (CHANNELS != 4 || chan_value(image2_p11, 3) != 0) begin sum2 = sum2 + chan_value(image2_p11, c); count2 = count2 + 1; end
            image1_out[c*DATA_WIDTH +: DATA_WIDTH] = (count1 == 0) ? {DATA_WIDTH{1'b0}} : sum1 / count1;
            image2_out[c*DATA_WIDTH +: DATA_WIDTH] = (count2 == 0) ? {DATA_WIDTH{1'b0}} : sum2 / count2;
        end
    end
end

endmodule
