# 30/10/24 -> Funcionant bé amb el modo CPU

import cv2
import numpy as np
import openvino.runtime as ov

# Inicializar el Core de OpenVINO
core = ov.Core()

# Cargar el modelo IR
model = core.read_model(r"C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\model_ir\saved_model.xml")

# Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
compiled_model = core.compile_model(model=model, device_name="CPU")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

# Prepara la imagen de entrada
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Cambia a 224x224
    img = img.astype(np.float32)  # Asegura el tipo de datos
    img = img / 255.0  # Normaliza los valores de píxeles (0-1)
    img = img[np.newaxis, :]  # Añadir dimensión de batch para que sea (1, 224, 224, 3)
    return img


#Info
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

# Realiza la inferencia
image_path = r"C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\tinyImageNet3K\00_prova_electricguitar.jpg"
input_image = preprocess_image(image_path)

# Inferencia
result = compiled_model(input_image)[output_key]  # Ejecuta la inferencia
result_index = np.argmax(result)


# Procesa la salida
print(result_index)  # Ajusta según el formato de salida de tu modelo
