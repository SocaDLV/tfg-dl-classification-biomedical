import onnx
import os

rutaModel = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/model2b.onnx')
model = onnx.load(rutaModel)
print(onnx.helper.printable_graph(model.graph))

