import torch
import os
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = r'C:\Users\...\best16-1-25.pt'

model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True)
model.eval()  


dummy_input = torch.rand(1, 3, 128, 128) 

torch.onnx.export(model, dummy_input, 'best16-1-25-Jetson_fix-128x128.onnx', opset_version=15)

print("Modelo exportado a ONNX con opset 15")

pathlib.PosixPath = temp