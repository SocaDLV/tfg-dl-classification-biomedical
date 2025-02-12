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
torch.onnx.export(model, (torch.rand(1, 3, 224, 224), ), 'best16-1-25.onnx')

# Revertir posixpath
pathlib.PosixPath = temp