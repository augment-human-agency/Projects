import time
from execution.cpu_runtime import CPUExecutor
from execution.fpga_sim_runtime import FPGAExecutor

class Profiler:
    def __init__(self, model_path):
        self.model_path = model_path

    def benchmark(self, input_data):
        cpu_executor = CPUExecutor(self.model_path)
        fpga_executor = FPGAExecutor(self.model_path)

        # CPU Profiling
        start_time = time.time()
        cpu_executor.run(input_data)
        cpu_time = time.time() - start_time

        # FPGA Profiling
        start_time = time.time()
        fpga_executor.run(input_data)
        fpga_time = time.time() - start_time

        print(f"🖥️ CPU Execution Time: {cpu_time:.6f} seconds")
        print(f"⚡ FPGA Execution Time (Simulated): {fpga_time:.6f} seconds")

# Example Usage
if __name__ == "__main__":
    profiler = Profiler("models/sample_cnn.onnx")
    profiler.benchmark([[1, 2, 3]])
