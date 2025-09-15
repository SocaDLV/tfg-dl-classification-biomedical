import onnx

rutaModel = r'C:\Users\...\model2b.onnx'
model = onnx.load(rutaModel)
print(onnx.helper.printable_graph(model.graph))