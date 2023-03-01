# pip install onnx-simplifier



from onnxsim import simplify
onnx_model = onnx.load(output_path)  # load onnx model
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_path)
print('finished exporting onnx')




