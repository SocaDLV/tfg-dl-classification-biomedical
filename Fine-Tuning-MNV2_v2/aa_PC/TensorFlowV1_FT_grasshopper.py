import os
import re
import time
import numpy as np
import psutil
import cv2
import tensorflow as tf
from pathlib import Path

# Cargar el archivo de clases de Tiny ImageNet (asegúrate de tener este archivo)
def load_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()[:2]
            labels[img_name] = label
    return labels

# Función para decodificar las predicciones de Tiny ImageNet
def decode_tiny_imagenet_predictions(predictions, top=1):
    # Archivo con las clases de Tiny ImageNet
    class_names = []
    with open(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\tinyimagenet200_classes.txt')) as f: 
        class_names = [line.strip() for line in f.readlines()]

    # Obtener las predicciones más altas
    top_indices = np.argsort(predictions[0])[::-1][:top]
    top_predictions = [(class_names[i], predictions[0][i]) for i in top_indices]

    return top_predictions

# Ruta de las etiquetas correctas por cada foto
labels_dict = load_labels(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\PC_validation_imgs_correct_preds.txt'))

# Ruta de las imágenes de prueba (dataset Tiny ImageNet)
image_folder = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\MobileNetV2\tinyImageNet3K_classGrasshopper')

# Función para ajustar las imágenes, comprueba que tengan todos los canales RGB, y las prepara para la inferencia
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)            # Asegura el tipo de datos
    img = img / 127.5 - 1.0                 # Normaliza los valores de píxeles (-1.0 a 1.0)
    img = img[np.newaxis, :]                # Añadir dimensión de batch para que sea (1, 224, 224, 3)
    return img

# Función para ordenar por número dentro del nombre del archivo
def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

# Función para medir el uso de CPU y tiempos de inferencia
def classify_and_measure(image_folder):
    # Asegúrate de cargar el modelo previamente entrenado y guardado
    model = tf.keras.models.load_model(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v2\v1_webMedium.com\modelosFTuneados\ft_mobilenetv2_tinyin_v8.h5'))  # Cargar el modelo fine-tuned
    model.summary(show_trainable=True, expand_nested=True)
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)       # Medir uso inicial de CPU, intervalo de 1 segundo para mayor precisión

    # Variables para el % de aciertos
    correct_predictions = 0
    total_images = 0

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.perf_counter()                   # Medir el tiempo de inferencia
        preds = model.predict(img)                         # Hacer la predicción
        inference_time = time.perf_counter() - start_time  # Calcular el tiempo de inferencia
        total_inference_time += inference_time

        # Decodificar la predicción usando la función personalizada
        decoded_preds = decode_tiny_imagenet_predictions(preds, top=1)
        print(f"Predicción para {img_file}: {decoded_preds}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        true_label = 'n02226429'  # Obtener etiqueta real
        print(decoded_preds[0][0])
        print(true_label)
        # Comparar predicción con etiqueta real
        if decoded_preds[0][0] == true_label:
            correct_predictions += 1
            print(f"La predicción es correcta! ✅")
        total_images += 1

    cpu_usage_end = psutil.cpu_percent(interval=1)         # Medir el uso del CPU al final, intervalo de 1 segundo para mayor precisión
    
    # Mostrar resultados de rendimiento
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

    # Calcular y mostrar precisión
    print(f"Se han acertado {correct_predictions} del total de {total_images}")
    accuracy = correct_predictions / total_images
    print(f"Precisión en las {total_images} imágenes: {accuracy * 100:.2f}%")

# Ejecutar la función de clasificación y medición
classify_and_measure(image_folder)
