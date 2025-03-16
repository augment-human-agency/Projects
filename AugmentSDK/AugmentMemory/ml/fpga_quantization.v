module fpga_quantization #(
    parameter INPUT_WIDTH = 32,
    parameter OUTPUT_WIDTH = 8  // INT8 execution
)(
    input clk,
    input [INPUT_WIDTH-1:0] input_data,
    output reg [OUTPUT_WIDTH-1:0] quantized_output
);

    always @(posedge clk) begin
        // Dynamic Scaling - If higher precision needed, keep INT16
        if (input_data > 32767) 
            quantized_output <= input_data >> 8; // Convert FP16 to INT8
        else 
            quantized_output <= input_data >> 4; // Convert INT16 to INT8
    end

endmodule
