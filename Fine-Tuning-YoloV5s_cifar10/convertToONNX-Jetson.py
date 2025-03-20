import torch
import os
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Ruta del modelo
model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-YoloV5s_cifar10\model\best16-1-25.pt'

# Cargar el modelo usando torch.load()
model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True)
model.eval()  # Coloca el modelo en modo evaluación

# Crear una entrada dummy con las dimensiones adecuadas (batch_size=1, 3 canales, 32x32)
dummy_input = torch.rand(1, 3, 32, 32)

# Exportar a ONNX con opset 15
torch.onnx.export(model, dummy_input, 'best16-1-25-Jetson.onnx', opset_version=15)

print("Modelo exportado a ONNX con opset 15")

pathlib.PosixPath = temp
