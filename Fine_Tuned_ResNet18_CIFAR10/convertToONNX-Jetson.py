import torch
import os

model_path = r'C:\Users\...\stage-2-pytorch'

model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  

dummy_input = torch.rand(1, 3, 32, 32)

torch.onnx.export(model, dummy_input, 'stage-2-Jetson.onnx', opset_version=15)

print("Modelo exportado a ONNX con opset 15")
