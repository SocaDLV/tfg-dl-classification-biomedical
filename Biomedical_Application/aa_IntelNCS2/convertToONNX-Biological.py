import torchvision
import torch
import os

model_path = r'C:\Users\...\biological-stage-2-pytorch'

model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

torch.onnx.export(model, (torch.rand(1, 3, 240, 320), ), 'biological-stage-2.onnx')