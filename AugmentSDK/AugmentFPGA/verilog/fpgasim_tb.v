`timescale 1ns / 1ps

module FPGA_SIM_TB;
    
    // Clock and reset signals
    reg clk;
    reg reset;

    // Input tensor (4 elements)
    reg [31:0] input_tensor [0:3];
    
    // Output tensor (from DUT - Device Under Test)
    wire [31:0] output_tensor [0:3];

    // Instantiate the FPGA_SIM_KERNEL
    FPGA_SIM_KERNEL uut (
        .clk(clk),
        .reset(reset),
        .input_tensor(input_tensor),
        .output_tensor(output_tensor)
    );

    // Clock Generation (50MHz example)
    always #10 clk = ~clk;

    integer i;

    // Test sequence
    initial begin
        // Initialize clock and reset
        clk = 0;
        reset = 1;
        #20 reset = 0; // Deassert reset after 20ns

        // Apply test input tensors
        input_tensor[0] = 3;
        input_tensor[1] = 1;
        input_tensor[2] = 4;
        input_tensor[3] = 2;

        #20; // Wait for computation

        // Display Results
        $display("Input Tensor: %d, %d, %d, %d", input_tensor[0], input_tensor[1], input_tensor[2], input_tensor[3]);
        $display("Output Tensor: %d, %d, %d, %d", output_tensor[0], output_tensor[1], output_tensor[2], output_tensor[3]);

        // Check results manually in simulator
        #50;
        $stop;
    end

endmodule
