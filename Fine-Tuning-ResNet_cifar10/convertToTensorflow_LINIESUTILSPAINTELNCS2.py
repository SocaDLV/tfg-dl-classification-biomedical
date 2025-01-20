#DEPRECATED: Pero mirar ref. 66 pa inspirarse pa convertir a model OpenVINO

import torch
import tensorflow as tf
import onnx
from onnx_tf.backend import prepare

# Paso 1: Cargar el modelo PyTorch
# Reemplaza 'model_architecture' por la definición del modelo que utilizaste
from torchvision.models import resnet18  # Importa tu arquitectura del modelo

# Cargar el modelo .pth
model = resnet18()
model.load_state_dict(torch.load(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\model\stage-1AntesDelUltimPas74%Accuracy.pth', map_location=torch.device('cpu')))
model.eval()

# Paso 2: Exportar el modelo a ONNX
dummy_input = torch.randn(1, 3, 32, 32)  # Ajusta el tamaño de entrada según tu modelo
onnx_file_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsONNX\stage-1.onnx'
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_file_path, 
    opset_version=11,
    input_names=["input"], 
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print(f"Modelo exportado a ONNX: {onnx_file_path}")

# Paso 3: Convertir de ONNX a TensorFlow
onnx_model = onnx.load(onnx_file_path)
tf_rep = prepare(onnx_model)

# Exportar el modelo como un SavedModel de TensorFlow
saved_model_dir = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsTF'
tf_rep.export_graph(saved_model_dir)
print(f"Modelo convertido a formato TensorFlow SavedModel en: {saved_model_dir}")

# Paso 4: Convertir el SavedModel a formato H5
converter = tf.keras.models.load_model(saved_model_dir)
h5_file_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsTF\stage-1.h5'
converter.save(h5_file_path)
print(f"Modelo guardado en formato H5: {h5_file_path}")