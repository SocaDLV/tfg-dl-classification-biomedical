# V2 -> Test de rendiment % d'acerts. Ajustos per passar de nº de classe (0-999 en ImageNet) a etiqueta. 
# També pasem a Dataset de validació de Tiny Imagenet


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
model = core.read_model(r"C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\model_ir\saved_model.xml")

# Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
compiled_model = core.compile_model(model=model, device_name="CPU")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

# Cargar archivo de etiquetas reales de Tiny ImageNet
def load_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()[:2]
            labels[img_name] = label
    return labels

# Ruta de las etiquetas correctas por cada foto
labels_dict = load_labels(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\IntelNCS2_RPZ2W_validation_imgs_correct_preds.txt'))

# Ruta de las imágenes de prueba (dataset ImageNet)
image_folder = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\tinyImageNet3K_validation')

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

#input_image = preprocess_image(image_path)

# Función para ordenar por número dentro del nombre del archivo
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)       # Medir uso inicial de CPU, intervalo de 1 segundo para mayor precisión

    # Variables para el % de aciertos
    correct_predictions = 0
    total_images = 0

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.perf_counter()                   
        result = compiled_model(img)[output_key]           # Ejecuta la inferencia
        inference_time = time.perf_counter() - start_time  # Tiempo de inferencia, V2 -> time.time per time.perf_counter
        total_inference_time += inference_time

        # Decodificar, mostrar la predicción y ver si es correcta
        result_index = np.argmax(result)
        print(f"Predicción para {img_file}: {result_index}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        true_label = labels_dict[os.path.basename(img_file)]  # Obtener etiqueta real
        #print(true_label)                                     # Comprobar que imprime bien la etiqueta real
        #print(result_index)                                   # Comprobar que imprime bien la etiqueta inferida

        # Comparar predicción con etiqueta
        if int(result_index) == int(true_label):
            correct_predictions += 1
            print(f"La predicción es correcta! ✅")
        total_images += 1
    

    cpu_usage_end = psutil.cpu_percent(interval=1)         # V2 -> Medir el uso del CPU al final, intervalo de 1 segundo para mayor precisión
    
    # Mostrar resultados de rendimiento
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

    # Calcular y mostrar precisión
    print(f"Se han acertado {correct_predictions} del total de {total_images}")
    accuracy = correct_predictions / total_images
    print(f"Precisión en las {total_images} imágenes: {accuracy * 100:.2f}%")

# Ejecutar la función de clasificación y medición
classify_and_measure(image_folder)