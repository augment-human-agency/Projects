module fpga_memory_controller #(
    parameter MEM_WIDTH = 256,  // DRAM data width (bits)
    parameter BURST_SIZE = 64,  // Burst read size
    parameter SEQ_LEN = 65536  // 64K tokens
)(
    input clk,
    input reset,
    input read_enable,
    input [31:0] addr_in,
    output reg [MEM_WIDTH-1:0] data_out
);

    reg [MEM_WIDTH-1:0] dram_mem [0:SEQ_LEN-1];
    integer i;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            data_out <= 0;
        end else if (read_enable) begin
            for (i = 0; i < BURST_SIZE; i = i + 1) begin
                data_out <= dram_mem[addr_in + i];
            end
        end
    end
endmodule
