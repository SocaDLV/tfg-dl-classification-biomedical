#V4 -> Inclou el test de % d'acerts. + Utilitza les imatges de "validation".

import os
import re
import time
import numpy as np
import psutil
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from pathlib import Path
from PIL import Image

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'   #Desactivar XLA pq genera algún error, avoltes.
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=true'    #Tindre-lo 'True' fa que vaja molt poc a poc.

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'                           # V2-> Desactivar búsqueda de GPU al sistema.

# Cargar el modelo preentrenado MobileNetV2 con pesos de ImageNet
model = MobileNetV2(weights='imagenet')

# Cargar archivo de etiquetas reales de Tiny ImageNet
def load_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()[:2]
            labels[img_name] = label
    return labels


# Ruta de las etiquetas correctas por cada foto
labels_dict = load_labels(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\PC_validation_imgs_correct_preds.txt'))

# Ruta de las imágenes de prueba (dataset ImageNet)
image_folder = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\tinyImageNet3K_validation')

# Función para redimensionar las imágenes a 224x224, comprueba que tengan todos los canales RGB, y las prepara para la inferencia
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Cambia a 224x224
    img = img.astype(np.float32)       # Asegura el tipo de datos
    img = img / 255.0                  # Normaliza los valores de píxeles (0-1)
    img = img[np.newaxis, :]           # Añadir dimensión de batch para que sea (1, 224, 224, 3)
    return img

# Función para ordenar por número dentro del nombre del archivo
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)       # V2 -> Medir uso inicial de CPU, intervalo de 1 segundo para mayor precisión

    # Variables para el % de aciertos
    correct_predictions = 0
    total_images = 0

    if total_images == 0:
        for img_file in image_files:
            counter = 0
            if counter == 0:
                img = cv2.imread(img_file)
                img = cv2.resize(img, (224, 224))  # Cambia a 224x224
                #img = img.astype(np.float32)       # Asegura el tipo de datos
                #img = img / 255.0                  # Normaliza los valores de píxeles (0-1)
                img = img[np.newaxis, :]           # Añadir dimensión de batch para que sea (1, 224, 224, 3)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convertir de BGR a RGB para matplotlib
                plt.title("Imagen procesada (224x224)")
                plt.axis('off')
                plt.show()
                counter = counter + 1

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.perf_counter()                   # V2 -> time.time per time.perf_counter
        preds = model.predict(img)                         # Hacer la predicción
        inference_time = time.perf_counter() - start_time  # Tiempo de inferencia, V2 -> time.time per time.perf_counter
        total_inference_time += inference_time

        # Decodificar, comparar con el resultado correcto y mostrar la predicción + si hemos acertados
        decoded_preds = decode_predictions(preds, top=1)[0]
        print(f"Predicción para {img_file}: {decoded_preds}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        true_label = labels_dict[os.path.basename(img_file)]  # Obtener etiqueta real
        #print(true_label)                                    # Comprobar que imprime bien la etiqueta real
        #print((decoded_preds[0])[0])                         # Comprobar que imprime bien la etiqueta inferida
        
        # Comparar predicción con etiqueta
        if (decoded_preds[0])[0] == true_label:
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