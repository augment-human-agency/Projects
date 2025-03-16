import onnxruntime as ort

class CPUExecutor:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = ort.InferenceSession(self.model_path)

    def run(self, input_data):
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name
        result = self.session.run([output_name], {input_name: input_data})
        return result

# Example Usage
if __name__ == "__main__":
    executor = CPUExecutor("models/sample_cnn.onnx")
    output = executor.run([[1, 2, 3]])
    print(f"CPU Output: {output}")
