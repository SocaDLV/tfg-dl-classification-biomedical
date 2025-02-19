import onnx

rutaModel = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\modelosFTuneados\model2b.onnx'
model = onnx.load(rutaModel)
print(onnx.helper.printable_graph(model.graph))

