import torchvision
import torch
import os

#model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsPytorch\stage-2-pytorch'

model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-ResNet_cifar10/modelsPytorch/stage-2-pytorch')

# Cargar el modelo usando torch.load()
model = torch.load(model_path, map_location=torch.device('cuda'))

# 1. Export to ONNX
torch.onnx.export(model, (torch.rand(1, 3, 32, 32), ), 'stage-2-cuda.onnx')
