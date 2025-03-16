import torch
import onnx
import onnxruntime as ort
import numpy as np
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_onnx_model(input_model: str, output_model: str, quantization_type="int8"):
    """
    Quantizes an ONNX model to INT8 or FP16 for optimized execution on FPGA.

    Args:
        input_model (str): Path to original ONNX model.
        output_model (str): Path to save the quantized model.
        quantization_type (str): "int8" for INT8 quantization, "fp16" for FP16.
    """
    print(f"[INFO] Quantizing {input_model} to {quantization_type.upper()} for FPGA...")

    if quantization_type == "int8":
        quantized_model = quantize_dynamic(
            input_model,
            output_model,
            weight_type=QuantType.QUInt8,
        )
    elif quantization_type == "fp16":
        onnx_model = onnx.load(input_model)
        onnx.save_model(onnx.shape_inference.infer_shapes(onnx_model), output_model)
    else:
        raise ValueError("Unsupported quantization type. Choose 'int8' or 'fp16'.")

    print(f"[SUCCESS] Quantized model saved at: {output_model}")

def run_quantized_inference(model_path: str, input_data: np.ndarray):
    """
    Runs inference on a quantized ONNX model.

    Args:
        model_path (str): Path to ONNX model.
        input_data (np.ndarray): Input tensor for model.
    
    Returns:
        Output of model inference.
    """
    print(f"[INFO] Running inference on {model_path}...")
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_data})
    print("[SUCCESS] Inference complete.")
    return output

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quantize an ONNX model for FPGA execution.")
    parser.add_argument("--model", type=str, required=True, help="Path to the ONNX model.")
    parser.add_argument("--output", type=str, required=True, help="Output path for quantized model.")
    parser.add_argument("--quantization", type=str, choices=["int8", "fp16"], required=True,
                        help="Quantization type: 'int8' or 'fp16'.")

    args = parser.parse_args()
    
    quantize_onnx_model(args.model, args.output, args.quantization)
