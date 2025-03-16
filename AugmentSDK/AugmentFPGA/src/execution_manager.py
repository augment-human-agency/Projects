import platform
from execution.cpu_runtime import CPUExecutor
from execution.fpga_sim_runtime import FPGAExecutor

class ExecutionManager:
    def __init__(self, model_path, use_fpga=False):
        self.model_path = model_path
        self.use_fpga = use_fpga

    def run(self, input_data):
        if self.use_fpga:
            print("⚡ Running model on FPGA Simulation...")
            executor = FPGAExecutor(self.model_path)
        else:
            print("⚡ Running model on CPU...")
            executor = CPUExecutor(self.model_path)
        
        return executor.run(input_data)

# Example Usage
if __name__ == "__main__":
    executor = ExecutionManager("models/mobilenet.onnx", use_fpga=False)
    output = executor.run([[1, 2, 3]])  # Example input tensor
    print(f"Output: {output}")
