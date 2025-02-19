import numpy as np
import onnxruntime
import os
from PIL import Image

def preprocess_image(image_path, img_size=224):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img).astype(np.float32)
    img_array = (img_array / 127.5) - 1.0  # Normalización
    img_array = np.transpose(img_array, (2, 0, 1))  # Transponer de HWC a CHW
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch
    return img_array

# Cargar modelo ONNX
rutaModel= os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/model2b.onnx')
session = onnxruntime.InferenceSession(rutaModel)

# Preprocesar la imagen
rutaImage = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test/airplane/aeroplane_s_000036_png.rf.89d566af5e94c17d9ae99896ce11a63a.jpg')
input_image = preprocess_image(rutaImage)
print("Processed image shape:", input_image.shape)  # Debería ser (1, 3, 224, 224)

# Obtener nombres de entrada y salida
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Ejecutar la inferencia
outputs = session.run([output_name], {input_name: input_image})
print("Model outputs:", outputs)

