import torch
import numpy as np

class Quantizer:
    def __init__(self, mode="INT8"):
        self.mode = mode

    def quantize(self, tensor):
        if self.mode == "INT8":
            return (tensor / 256).to(torch.int8)  # Scale FP16 -> INT8
        elif self.mode == "INT16":
            return (tensor / 16).to(torch.int16)  # Scale FP16 -> INT16
        else:
            return tensor  # Keep as FP16

    def process_tensor(self, tensor):
        quantized = self.quantize(tensor)
        return quantized.numpy()

if __name__ == "__main__":
    tensor = torch.randn(1, 512)
    quantizer = Quantizer(mode="INT8")  # Switch to INT16 if needed
    result = quantizer.process_tensor(tensor)
    print("Quantized Tensor:", result)
