import os
import time
import numpy as np
import psutil

import tensorflow as tf
#from tensorflow.python.keras.models import MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from pathlib import Path
from PIL import Image

#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'   #Desactivar XLA pq genera algún error, avoltes.
#os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=true'    #Tindre-lo 'True' fa que vaja molt poc a poc.

# Cargar el modelo preentrenado MobileNetV2 con pesos de ImageNet
model = MobileNetV2(weights='imagenet')

# Ruta de las imágenes de prueba (dataset ImageNet)
image_folder = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\tinyImageNet')

# Función para redimensionar las imágenes a 224x224, comprueba que tengan todos los canales RGB, y las prepara para la inferencia
def preprocess_image(image_path):
    # Cargar la imagen
    img = image.load_img(image_path, target_size=(224,224))
    
    # Convertir la imagen a un array numpy
    img_array = image.img_to_array(img)
    
    # Verificar el número de canales (última dimensión)
    if img_array.shape[-1] == 1:
        # Si la imagen tiene 1 canal (escala de grises), la replicamos 3 veces
        img_array = np.repeat(img_array, 3, axis=-1)
    
    # Preprocesar la imagen (normalización)
    img_array = preprocess_input(img_array)
    
    # Añadir una dimensión más (batch_size, altura, ancho, canales)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=None)  # Medir el uso inicial del CPU

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.time()
        preds = model.predict(img)  # Hacer la predicción
        inference_time = time.time() - start_time  # Tiempo de inferencia
        total_inference_time += inference_time

        # Decodificar y mostrar la predicción
        decoded_preds = decode_predictions(preds, top=3)[0]
        print(f"Predicción para {img_file}: {decoded_preds}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    
    cpu_usage_end = psutil.cpu_percent(interval=None)  # Medir el uso del CPU al final
    
    # Mostrar resultados de rendimiento
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

# Ejecutar la función de clasificación y medición
classify_and_measure(image_folder)