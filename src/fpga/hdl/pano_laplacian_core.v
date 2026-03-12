module pano_laplacian_core #(
    parameter CHANNELS = 4,
    parameter DATA_WIDTH = 18
) (
    input wire x_odd,
    input wire y_odd,
    input wire [CHANNELS*DATA_WIDTH-1:0] high_pixel,
    input wire [CHANNELS*DATA_WIDTH-1:0] low00,
    input wire [CHANNELS*DATA_WIDTH-1:0] low10,
    input wire [CHANNELS*DATA_WIDTH-1:0] low01,
    input wire [CHANNELS*DATA_WIDTH-1:0] low11,
    output reg [CHANNELS*DATA_WIDTH-1:0] laplacian_pixel
);

integer c;
integer weighted_sum;
integer weight_sum;
integer upsampled;
integer w00;
integer w10;
integer w01;
integer w11;
integer alpha_index;
localparam signed [31:0] MAX_VALUE = (1 << (DATA_WIDTH - 1)) - 1;
localparam signed [31:0] MIN_VALUE = -(1 << (DATA_WIDTH - 1));

function signed [DATA_WIDTH-1:0] chan_value;
    input [CHANNELS*DATA_WIDTH-1:0] pixel;
    input integer index;
    begin
        chan_value = pixel[index*DATA_WIDTH +: DATA_WIDTH];
    end
endfunction

function signed [DATA_WIDTH-1:0] sat_value;
    input integer value;
    begin
        if (value > MAX_VALUE) begin
            sat_value = MAX_VALUE[DATA_WIDTH-1:0];
        end else if (value < MIN_VALUE) begin
            sat_value = MIN_VALUE[DATA_WIDTH-1:0];
        end else begin
            sat_value = value[DATA_WIDTH-1:0];
        end
    end
endfunction

always @(*) begin
    alpha_index = CHANNELS - 1;
    if (!x_odd && !y_odd) begin
        w00 = 4; w10 = 0; w01 = 0; w11 = 0;
    end else if (x_odd && !y_odd) begin
        w00 = 2; w10 = 2; w01 = 0; w11 = 0;
    end else if (!x_odd && y_odd) begin
        w00 = 2; w10 = 0; w01 = 2; w11 = 0;
    end else begin
        w00 = 1; w10 = 1; w01 = 1; w11 = 1;
    end

    laplacian_pixel = {CHANNELS*DATA_WIDTH{1'b0}};
    for (c = 0; c < CHANNELS; c = c + 1) begin
        if (CHANNELS == 4 && c == alpha_index) begin
            laplacian_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(high_pixel, c);
        end else begin
            weighted_sum = 0;
            weight_sum = 0;
            if (CHANNELS != 4 || chan_value(low00, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(low00, c) * w00); weight_sum = weight_sum + w00; end
            if (CHANNELS != 4 || chan_value(low10, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(low10, c) * w10); weight_sum = weight_sum + w10; end
            if (CHANNELS != 4 || chan_value(low01, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(low01, c) * w01); weight_sum = weight_sum + w01; end
            if (CHANNELS != 4 || chan_value(low11, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(low11, c) * w11); weight_sum = weight_sum + w11; end
            upsampled = (weight_sum == 0) ? 0 : (weighted_sum / weight_sum);
            laplacian_pixel[c*DATA_WIDTH +: DATA_WIDTH] = sat_value(chan_value(high_pixel, c) - upsampled);
        end
    end
end

endmodule
