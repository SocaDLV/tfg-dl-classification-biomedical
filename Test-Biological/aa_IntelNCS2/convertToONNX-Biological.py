import torchvision
import torch
import os

#model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsPytorch\stage-2-pytorch'

model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Test-Biological\models\biological-stage-2-pytorch'

# Cargar el modelo usando torch.load()
model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

# Exportar a ONNX
torch.onnx.export(model, (torch.rand(1, 3, 240, 320), ), 'biological-stage-2.onnx')
