module pano_kernel_dispatch (
    input wire clk,
    input wire rstn,
    input wire soft_reset_pulse,
    input wire start_pulse,
    input wire done_ack,
    input wire [31:0] opcode,
    input wire irq_enable,
    input wire remap_done,
    input wire copy_done,
    input wire downsample_done,
    input wire laplacian_done,
    input wire blend_done,
    input wire reconstruct_done,
    output reg remap_start,
    output reg copy_start,
    output reg downsample_start,
    output reg laplacian_start,
    output reg blend_start,
    output reg reconstruct_start,
    output reg busy,
    output reg done,
    output reg error,
    output wire irq
);

localparam OP_REMAP = 32'd1;
localparam OP_COPY = 32'd2;
localparam OP_PYRAMID_BLEND = 32'd3;
localparam OP_DOWNSAMPLE = 32'd4;
localparam OP_LAPLACIAN = 32'd5;
localparam OP_BLEND = 32'd6;
localparam OP_RECONSTRUCT = 32'd7;

reg [31:0] active_opcode;

assign irq = irq_enable && done;

always @(posedge clk or negedge rstn) begin
    if (!rstn) begin
        active_opcode <= 32'd0;
        remap_start <= 1'b0;
        copy_start <= 1'b0;
        downsample_start <= 1'b0;
        laplacian_start <= 1'b0;
        blend_start <= 1'b0;
        reconstruct_start <= 1'b0;
        busy <= 1'b0;
        done <= 1'b0;
        error <= 1'b0;
    end else begin
        remap_start <= 1'b0;
        copy_start <= 1'b0;
        downsample_start <= 1'b0;
        laplacian_start <= 1'b0;
        blend_start <= 1'b0;
        reconstruct_start <= 1'b0;

        if (soft_reset_pulse) begin
            active_opcode <= 32'd0;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
        end else if (done_ack) begin
            done <= 1'b0;
            error <= 1'b0;
        end else if (!busy && start_pulse) begin
            active_opcode <= opcode;
            busy <= 1'b1;
            done <= 1'b0;
            error <= 1'b0;
            case (opcode)
                OP_REMAP: remap_start <= 1'b1;
                OP_COPY: copy_start <= 1'b1;
                OP_PYRAMID_BLEND,
                OP_BLEND: blend_start <= 1'b1;
                OP_DOWNSAMPLE: downsample_start <= 1'b1;
                OP_LAPLACIAN: laplacian_start <= 1'b1;
                OP_RECONSTRUCT: reconstruct_start <= 1'b1;
                default: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                end
            endcase
        end else if (busy) begin
            case (active_opcode)
                OP_REMAP: if (remap_done) begin busy <= 1'b0; done <= 1'b1; end
                OP_COPY: if (copy_done) begin busy <= 1'b0; done <= 1'b1; end
                OP_PYRAMID_BLEND,
                OP_BLEND: if (blend_done) begin busy <= 1'b0; done <= 1'b1; end
                OP_DOWNSAMPLE: if (downsample_done) begin busy <= 1'b0; done <= 1'b1; end
                OP_LAPLACIAN: if (laplacian_done) begin busy <= 1'b0; done <= 1'b1; end
                OP_RECONSTRUCT: if (reconstruct_done) begin busy <= 1'b0; done <= 1'b1; end
                default: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                end
            endcase
        end
    end
end

endmodule
