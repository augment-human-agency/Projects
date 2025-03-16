module multihead_attention #(
    parameter HEADS = 4,  // Number of attention heads
    parameter DIM = 4     // Feature dimensions per head
)(
    input clk,
    input reset,
    input [31:0] query [0:HEADS-1][0:DIM-1],
    input [31:0] key [0:HEADS-1][0:DIM-1],
    input [31:0] value [0:HEADS-1][0:DIM-1],
    output reg [31:0] attention_output [0:HEADS-1][0:DIM-1]
);

    reg [31:0] scores [0:HEADS-1][0:DIM-1];
    reg [31:0] sum [0:HEADS-1];
    integer i, j;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < HEADS; i = i + 1) begin
                sum[i] = 0;
                for (j = 0; j < DIM; j = j + 1) begin
                    scores[i][j] = 0;
                    attention_output[i][j] = 0;
                end
            end
        end else begin
            for (i = 0; i < HEADS; i = i + 1) begin
                sum[i] = 0;
                for (j = 0; j < DIM; j = j + 1) begin
                    scores[i][j] = query[i][j] * key[i][j]; // Dot-product attention
                    sum[i] = sum[i] + scores[i][j];
                end
                for (j = 0; j < DIM; j = j + 1) begin
                    attention_output[i][j] = (scores[i][j] * value[i][j]) / sum[i]; // Weighted sum
                end
            end
        end
    end
endmodule
