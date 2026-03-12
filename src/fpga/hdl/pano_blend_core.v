module pano_blend_core #(
    parameter CHANNELS = 4,
    parameter DATA_WIDTH = 18,
    parameter MASK_WIDTH = 16
) (
    input wire [CHANNELS*DATA_WIDTH-1:0] lap1_pixel,
    input wire [CHANNELS*DATA_WIDTH-1:0] lap2_pixel,
    input wire [MASK_WIDTH-1:0] mask_weight,
    output reg [CHANNELS*DATA_WIDTH-1:0] blended_pixel
);

integer c;
integer alpha_index;
reg signed [47:0] blend_value;
localparam integer MASK_ONE = (1 << MASK_WIDTH) - 1;
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
    input signed [47:0] value;
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
    blended_pixel = {CHANNELS*DATA_WIDTH{1'b0}};
    alpha_index = CHANNELS - 1;
    for (c = 0; c < CHANNELS; c = c + 1) begin
        if (CHANNELS == 4 && c == alpha_index) begin
            if (chan_value(lap1_pixel, alpha_index) == 0) begin
                blended_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(lap2_pixel, alpha_index);
            end else begin
                blended_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(lap1_pixel, alpha_index);
            end
        end else if (CHANNELS == 4 && chan_value(lap1_pixel, alpha_index) == 0) begin
            blended_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(lap2_pixel, c);
        end else if (CHANNELS == 4 && chan_value(lap2_pixel, alpha_index) == 0) begin
            blended_pixel[c*DATA_WIDTH +: DATA_WIDTH] = chan_value(lap1_pixel, c);
        end else begin
            blend_value =
                ($signed(chan_value(lap1_pixel, c)) * $signed({1'b0, mask_weight})) +
                ($signed(chan_value(lap2_pixel, c)) * $signed({1'b0, (MASK_ONE - mask_weight)}));
            blended_pixel[c*DATA_WIDTH +: DATA_WIDTH] = sat_value(blend_value / MASK_ONE);
        end
    end
end

endmodule
