import torch
import os

# Ruta del modelo
model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-ResNet_cifar10/modelsPytorch/stage-2-pytorch')

# Cargar el modelo usando torch.load()
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # Coloca el modelo en modo evaluación

# Crear una entrada dummy con las dimensiones adecuadas (batch_size=1, 3 canales, 32x32)
dummy_input = torch.rand(1, 3, 32, 32)

# Exportar a ONNX con opset 15
torch.onnx.export(model, dummy_input, 'stage-2-Jetson.onnx', opset_version=15)

print("Modelo exportado a ONNX con opset 15")
