module pano_reconstruct_core #(
    parameter CHANNELS = 4,
    parameter DATA_WIDTH = 18
) (
    input wire x_odd,
    input wire y_odd,
    input wire [CHANNELS*DATA_WIDTH-1:0] lower00,
    input wire [CHANNELS*DATA_WIDTH-1:0] lower10,
    input wire [CHANNELS*DATA_WIDTH-1:0] lower01,
    input wire [CHANNELS*DATA_WIDTH-1:0] lower11,
    input wire [CHANNELS*DATA_WIDTH-1:0] lap_pixel,
    output reg [CHANNELS*DATA_WIDTH-1:0] recon_pixel
);

integer c;
integer weighted_sum;
integer weight_sum;
integer upsampled;
integer alpha_index;
integer w00;
integer w10;
integer w01;
integer w11;
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

    recon_pixel = {CHANNELS*DATA_WIDTH{1'b0}};
    for (c = 0; c < CHANNELS; c = c + 1) begin
        if (CHANNELS == 4 && c == alpha_index) begin
            recon_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(lap_pixel, c);
        end else begin
            weighted_sum = 0;
            weight_sum = 0;
            if (CHANNELS != 4 || chan_value(lower00, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(lower00, c) * w00); weight_sum = weight_sum + w00; end
            if (CHANNELS != 4 || chan_value(lower10, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(lower10, c) * w10); weight_sum = weight_sum + w10; end
            if (CHANNELS != 4 || chan_value(lower01, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(lower01, c) * w01); weight_sum = weight_sum + w01; end
            if (CHANNELS != 4 || chan_value(lower11, alpha_index) != 0) begin weighted_sum = weighted_sum + (chan_value(lower11, c) * w11); weight_sum = weight_sum + w11; end
            upsampled = (weight_sum == 0) ? 0 : (weighted_sum / weight_sum);
            recon_pixel[c*DATA_WIDTH +: DATA_WIDTH] = sat_value(upsampled + chan_value(lap_pixel, c));
        end
    end
end

endmodule
