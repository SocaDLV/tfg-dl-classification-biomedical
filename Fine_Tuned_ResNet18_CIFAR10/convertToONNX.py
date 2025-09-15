import torchvision
import torch
import os

model_path = os.path.expanduser('~/codi/.../stage-2-pytorch')

model = torch.load(model_path, map_location=torch.device('cpu'))

torch.onnx.export(model, (torch.rand(1, 3, 32, 32), ), 'stage-2.onnx')