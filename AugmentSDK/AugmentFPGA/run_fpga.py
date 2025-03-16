import onnxruntime as ort
import numpy as np
import argparse
from src.onnx_fpga_mapper import map_onnx_to_fpga
from src.fpga_execution_manager import FPGAExecutionManager
from src.cpu_runtime import run_model_on_cpu

# Argument parser for CLI execution
parser = argparse.ArgumentParser(description="Run AI inference on FPGA or CPU fallback.")
parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file (numpy format).")
parser.add_argument("--fpga", action="store_true", help="Enable FPGA execution")
args = parser.parse_args()

# Load input tensor
input_tensor = np.load(args.input)

# Initialize FPGA execution manager
fpga_manager = FPGAExecutionManager()

def run_inference():
    if args.fpga:
        print("[INFO] Attempting to run model on FPGA...")
        try:
            mapped_model = map_onnx_to_fpga(args.model)
            output = fpga_manager.execute(mapped_model, input_tensor)
            print("[SUCCESS] Inference completed on FPGA.")
            return output
        except Exception as e:
            print(f"[WARNING] FPGA execution failed: {e}")
            print("[INFO] Falling back to CPU execution.")
    
    # Fallback: Run on CPU
    output = run_model_on_cpu(args.model, input_tensor)
    print("[INFO] Inference completed on CPU.")
    return output

# Run inference
if __name__ == "__main__":
    result = run_inference()
    print("[RESULT] Output:", result)
