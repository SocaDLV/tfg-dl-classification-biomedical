import torchvision
import torch
import pathlib
from pathlib import Path

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

model_path = r'C:\Users\...\best16-1-25'

model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True) 

torch.onnx.export(model, (torch.rand(1, 3, 128, 128), ), 'best16-1-25-128x128.onnx') 

pathlib.PosixPath = temp