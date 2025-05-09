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
dummy_input = torch.rand(1, 3, 128, 128) # FIX 09/05/2025 (basat en el de IntelNCS2): 1,3,128,128 -> Si no, dona error al executar en IntelNCS2. Expected era 228, solucionat canviant esta instrucció

# Exportar a ONNX con opset 15
torch.onnx.export(model, dummy_input, 'best16-1-25-Jetson_fix-128x128.onnx', opset_version=15)

print("Modelo exportado a ONNX con opset 15")

pathlib.PosixPath = temp
