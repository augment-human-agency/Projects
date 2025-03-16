import onnxruntime as ort
import numpy as np
import argparse
from src.cpu_runtime import run_model_on_cpu

# Argument parser for CLI execution
parser = argparse.ArgumentParser(description="Run AI inference on CPU.")
parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
parser.add_argument("--input", type=str, required=True, help="Path to the input tensor file (numpy format).")
args = parser.parse_args()

# Load input tensor
input_tensor = np.load(args.input)

def run_inference():
    print("[INFO] Running model on CPU...")
    output = run_model_on_cpu(args.model, input_tensor)
    print("[SUCCESS] Inference completed on CPU.")
    return output

# Run inference
if __name__ == "__main__":
    result = run_inference()
    print("[RESULT] Output:", result)
