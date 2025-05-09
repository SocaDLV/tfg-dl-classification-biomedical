import torchvision
import torch
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-YoloV5s_cifar10\model\best16-1-25'

# Cargar el modelo usando torch.load()
# Cargar modelo custom de yolov5 de ultralytics
model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True) #, map_location=torch.device('cpu'))

# Export to ONNX
torch.onnx.export(model, (torch.rand(1, 3, 128, 128), ), 'best16-1-25-128x128.onnx') # FIX 09/05/2025: 1,3,128,128 -> Si no, dona error al executar en IntelNCS2. Expected era 228, solucionat canviant esta instrucció

# Revertir posixpath
pathlib.PosixPath = temp