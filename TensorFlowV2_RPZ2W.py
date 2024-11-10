# v2 -> Test de % d'acerts + canvi a Dataset de Validació

import os
import re
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
from pathlib import Path

# Cargar el modelo TensorFlow Lite en la Raspberry Pi
model_path = "mobilenet_v2_optimized.tflite"
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Cargar archivo de etiquetas reales de Tiny ImageNet
def load_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()[:2]
            labels[img_name] = label
    return labels

# Ruta de las etiquetas correctas por cada foto
labels_dict = load_labels(Path(os.path.expanduser('~/codi/TFG/IntelNCS2_RPZ2W_validation_imgs_correct_preds.txt')))

# Función para redimensionar las imágenes a 224x224, comprobando canales RGB, y preparando la inferencia
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Función para ordenar por número dentro del nombre del archivo
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)

    # Variables para el % de aciertos
    correct_predictions = 0
    total_images = 0

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        # Preparar entrada para TensorFlow Lite
        interpreter.set_tensor(input_details[0]['index'], img)
        start_time = time.perf_counter()
        interpreter.invoke()                                  # Ejecutar la inferencia en TensorFlow Lite
        inference_time = time.perf_counter() - start_time
        total_inference_time += inference_time

        # Obtener predicción y ver si es correcta
        preds = interpreter.get_tensor(output_details[0]['index'])
        top_pred_idx = np.argmax(preds, axis=-1)
        print(f"Predicción para {img_file}: {top_pred_idx}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        true_label = labels_dict[os.path.basename(img_file)]  # Obtener etiqueta real
        print(true_label)                                     # Comprobar que imprime bien la etiqueta real
        print(int(top_pred_idx)[0])                           # Comprobar que imprime bien la etiqueta inferida

        # Comparar predicción con etiqueta
        if int(top_pred_idx)[0] == int(true_label):
            correct_predictions += 1
            print(f"La predicción es correcta! ✅")
        total_images += 1
    
    cpu_usage_end = psutil.cpu_percent(interval=1)
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

    # Calcular y mostrar precisión
    print(f"Se han acertado {correct_predictions} del total de {total_images}")
    accuracy = correct_predictions / total_images
    print(f"Precisión en las {total_images} imágenes: {accuracy * 100:.2f}%")

# Ejecutar la función de clasificación y medición
image_folder = Path(os.path.expanduser('~/codi/TFG/tinyImageNet3K_validation'))
classify_and_measure(image_folder)
