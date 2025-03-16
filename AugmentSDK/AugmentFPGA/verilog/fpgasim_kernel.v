module FPGA_SIM_KERNEL (
    input clk,                      // FPGA clock
    input reset,                    // Reset signal
    input [31:0] input_tensor [0:3], // 4-element input tensor (Example)
    output reg [31:0] output_tensor [0:3] // 4-element output tensor
);

    // Internal registers for computation
    reg [31:0] weight_matrix [0:3][0:3];  // Simulated AI weights
    reg [31:0] temp_result [0:3];

    integer i, j;

    // Initialize weight matrix (Simulated AI Model)
    initial begin
        weight_matrix[0][0] = 2; weight_matrix[0][1] = 1; weight_matrix[0][2] = 3; weight_matrix[0][3] = 0;
        weight_matrix[1][0] = 0; weight_matrix[1][1] = 1; weight_matrix[1][2] = 1; weight_matrix[1][3] = 2;
        weight_matrix[2][0] = 1; weight_matrix[2][1] = 2; weight_matrix[2][2] = 0; weight_matrix[2][3] = 1;
        weight_matrix[3][0] = 3; weight_matrix[3][1] = 0; weight_matrix[3][2] = 2; weight_matrix[3][3] = 1;
    end

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            for (i = 0; i < 4; i = i + 1) begin
                output_tensor[i] <= 0;
            end
        end else begin
            // Matrix Multiplication: input_tensor * weight_matrix
            for (i = 0; i < 4; i = i + 1) begin
                temp_result[i] = 0;
                for (j = 0; j < 4; j = j + 1) begin
                    temp_result[i] = temp_result[i] + (input_tensor[j] * weight_matrix[i][j]);
                end
                output_tensor[i] <= temp_result[i]; // Store result
            end
        end
    end

endmodule
