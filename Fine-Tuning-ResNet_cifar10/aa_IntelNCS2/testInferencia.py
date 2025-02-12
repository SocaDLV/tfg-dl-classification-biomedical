# Extremadament lento i poc precis (12%) en device_name="CPU"
# Molt ràpid (3000 imàgens en 30 segons) pero igual de imprecís en device_name="MYRIAD"

import os
import time
import numpy as np
import psutil
import cv2
import openvino.runtime as ov
from pathlib import Path
from random import sample


# Inicializar el Core de OpenVINO
core = ov.Core()

# Cargar el modelo ONNX
model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-ResNet_cifar10/modelsONNX/stage-2.onnx')
model = core.read_model(model_path)

# Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
compiled_model = core.compile_model(model=model, device_name="MYRIAD")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

# Definir las clases de CIFAR-10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Ruta a las imágenes de test
data_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test')

# Tamaño de imagen esperado por ResNet18 -> No ho utilitzem, les imatges ja estan a 32x32
IMG_SIZE = 32

# Cargar y procesar imágenes
def load_images(data_dir, total_images):
    data_dir = Path(data_dir).expanduser()
    image_files = list(data_dir.rglob('*.jpg'))  # Busca en subdirectorios
    
    if len(image_files) == 0:
        raise ValueError("No se encontraron imágenes en el directorio especificado.")
    
    # Ajusta el tamaño de la muestra al número disponible
    total_images = min(len(image_files), total_images)
    sampled_files = sample(image_files, total_images)
    
    return sampled_files

# Cargar las primeras 3000 imágenes
total_images = 3000
image_files = load_images(data_dir, total_images)

# Inicializar variables para métricas
correct_predictions = 0
total_inference_time = 0

# Necesari pal model
input_key = compiled_model.input(0)
output_key = compiled_model.output(0)
network_input_shape = input_key.shape

# Realizar inferencia sobre las imágenes en batches de tamaño 1
for i, file in enumerate(image_files):
    # Leer la imagen y obtener la etiqueta
    label = cifar10_classes.index(file.parent.name)  # Usar el nombre del subdirectorio como etiqueta
    image = cv2.imread(str(file))
    image = image.astype(np.float32)
    image = (image / 127.5) - 1
    #image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Añadir dimensión de batch
    image = np.expand_dims(image, axis=0)

    start_time = time.time()  # Tiempo de inicio para esta imagen

    result = compiled_model(image)[output_key]

    inference_time = time.time() - start_time
    total_inference_time += inference_time

    # Obtener predicción
    result_index = np.argmax(result)
    print(f"Predicción para imagen {i}: {result_index}")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    print(f"Etiqueta real: {cifar10_classes[label]}")

    # Contar los aciertos
    if result_index == label:
        correct_predictions += 1
        print("✅ Acierto")

# Calcular el porcentaje de aciertos
accuracy = (correct_predictions / total_images) * 100

# Mostrar resultados
print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")
