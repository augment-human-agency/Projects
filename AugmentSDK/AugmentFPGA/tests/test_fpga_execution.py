import time
import argparse
import numpy as np
from src.fpga_execution_manager import run_model_on_fpga
from src.cpu_runtime import run_model_on_cpu

# Argument parser for CLI execution
parser = argparse.ArgumentParser(description="Test AI inference on FPGA and compare with CPU.")
parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file (numpy format).")
parser.add_argument("--compare_cpu", action="store_true", help="Compare FPGA execution against CPU.")

args = parser.parse_args()

# Load input tensor
input_tensor = np.load(args.input)

def benchmark_fpga():
    print("[INFO] Running inference on FPGA...")
    start_time = time.time()
    output_fpga = run_model_on_fpga(args.model, input_tensor)
    fpga_time = time.time() - start_time
    print(f"[SUCCESS] FPGA execution completed in {fpga_time:.4f} seconds.")
    return output_fpga, fpga_time

def benchmark_cpu():
    print("[INFO] Running inference on CPU for comparison...")
    start_time = time.time()
    output_cpu = run_model_on_cpu(args.model, input_tensor)
    cpu_time = time.time() - start_time
    print(f"[SUCCESS] CPU execution completed in {cpu_time:.4f} seconds.")
    return output_cpu, cpu_time

# Run tests
if __name__ == "__main__":
    output_fpga, fpga_time = benchmark_fpga()
    
    if args.compare_cpu:
        output_cpu, cpu_time = benchmark_cpu()
        print("\n🔹 **Performance Comparison**")
        print(f"✅ FPGA Execution Time: {fpga_time:.4f} seconds")
        print(f"✅ CPU Execution Time: {cpu_time:.4f} seconds")
        speedup = cpu_time / fpga_time if fpga_time > 0 else float("inf")
        print(f"📈 Speedup Factor: {speedup:.2f}x faster on FPGA")
        
        # Optional: Validate Output Consistency
        if np.allclose(output_fpga, output_cpu, atol=1e-5):
            print("✅ Model output is consistent across CPU and FPGA.")
        else:
            print("⚠️ Warning: Output discrepancy detected between CPU and FPGA.")

    print("\n🎯 FPGA execution test completed.")
