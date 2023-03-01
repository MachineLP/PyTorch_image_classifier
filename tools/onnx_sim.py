# pip install onnx-simplifier


import onnx
from onnxsim import simplify

onnx_path = ""
output_path = ""
onnx_model = onnx.load(onnx_path)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')




