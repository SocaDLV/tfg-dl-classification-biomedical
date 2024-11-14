# Inclou les millores d'optimització de la V2 de PC

import os
import re
import time
import numpy as np
import psutil

import cv2
import numpy as np
import openvino.runtime as ov

from pathlib import Path
from PIL import Image

# Inicializar el Core de OpenVINO
core = ov.Core()

# Cargar el modelo IR
model = core.read_model(r"C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\model_ir\saved_model.xml")

# Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
compiled_model = core.compile_model(model=model, device_name="CPU")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

# Prepara la imagen de entrada
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Cambia a 224x224
    img = img.astype(np.float32)       # Asegura el tipo de datos
    img = img / 255.0                  # Normaliza los valores de píxeles (0-1)
    img = img[np.newaxis, :]           # Añadir dimensión de batch para que sea (1, 224, 224, 3)
    return img


#Info + Path fotos
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

#output_name = compiled_model.output(1) -> Vore si funciona!!!

network_input_shape = input_key.shape

image_folder = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\tinyImageNet3K')
#input_image = preprocess_image(image_path)

# Función para ordenar por número dentro del nombre del archivo
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)       # Medir uso inicial de CPU, intervalo de 1 segundo para mayor precisión

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.perf_counter()                   
        result = compiled_model(img)[output_key]           # Ejecuta la inferencia
        inference_time = time.perf_counter() - start_time  # Tiempo de inferencia, V2 -> time.time per time.perf_counter
        total_inference_time += inference_time

        # Decodificar y mostrar la predicción
        result_index = np.argmax(result)
        print(f"Predicción para {img_file}: {result_index}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    
    cpu_usage_end = psutil.cpu_percent(interval=1)         # V2 -> Medir el uso del CPU al final, intervalo de 1 segundo para mayor precisión
    
    # Mostrar resultados de rendimiento
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

# Ejecutar la función de clasificación y medición
classify_and_measure(image_folder)