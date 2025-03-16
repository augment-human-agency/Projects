import time
import argparse
import numpy as np
import onnx
import onnxruntime as ort
from src.quantization_utils import quantize_model

# Argument parser for CLI execution
parser = argparse.ArgumentParser(description="Test AI inference with different quantization levels on FPGA.")
parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file (numpy format).")
parser.add_argument("--quantization", type=str, choices=["fp32", "fp16", "int8"], required=True,
                    help="Quantization level: fp32 (full precision), fp16 (half precision), or int8 (integer quantization).")

args = parser.parse_args()

# Load input tensor
input_tensor = np.load(args.input)

def run_inference(model_path):
    """Runs inference using ONNX Runtime and measures execution time."""
    print(f"[INFO] Running inference on {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    start_time = time.time()
    output = session.run(None, {input_name: input_tensor})
    exec_time = time.time() - start_time
    print(f"[SUCCESS] Execution completed in {exec_time:.4f} seconds.\n")
    return output, exec_time

# Main execution
if __name__ == "__main__":
    print(f"\n🔹 Testing Model: {args.model}")
    print(f"🔹 Quantization Level: {args.quantization.upper()}")

    if args.quantization == "fp32":
        model_path = args.model
    else:
        quantized_model_path = f"models/quantized_{args.quantization}.onnx"
        quantize_model(args.model, quantized_model_path, args.quantization)
        model_path = quantized_model_path

    output, exec_time = run_inference(model_path)

    print(f"✅ Final Execution Time ({args.quantization.upper()}): {exec_time:.4f} seconds")
    print("🎯 Quantization test completed successfully.")
