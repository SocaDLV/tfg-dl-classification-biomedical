# basado en el v1 pero sin utilizar tensorflow_datasets ya que requiere de tensorflow

import os
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
from pathlib import Path
from random import sample

# Ruta al modelo TensorFlow Lite
model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/mnv2_cifar10_bo_optimized.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir las clases de CIFAR-10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# Ruta a las imágenes de test
data_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test')

# Tamaño de imagen esperado por MobileNetV2
IMG_SIZE = 224

# Cargar y procesar imágenes
def load_images(data_dir, total_images):
    data_dir = Path(data_dir).expanduser()
    image_files = list(data_dir.rglob('*.jpg'))  # Busca en subdirectorios
    
    if len(image_files) == 0:
        raise ValueError("No se encontraron imágenes en el directorio especificado.")

    # Ajusta el tamaño de la muestra al número disponible
    total_images = min(len(image_files), total_images)
    sampled_files = sample(image_files, total_images)
    
    images, labels = [], []
    for file in sampled_files:
        label = file.parent.name  # Usa el nombre del subdirectorio como etiqueta
        image = cv2.imread(str(file))
        image = image.astype(np.float32)
        image = (image / 127.5) - 1
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        images.append(image)
        labels.append(cifar10_classes.index(label))
    
    return np.array(images), np.array(labels)

# Cargar las primeras 3000 imágenes
total_images = 3000
images, labels = load_images(data_dir, total_images)

# Inicializar variables para métricas
correct_predictions = 0
total_inference_time = 0

# Realizar inferencia sobre las imágenes
for i in range(total_images):
    image = np.expand_dims(images[i], axis=0)  # Añadir dimensión de batch
    label = labels[i]

    start_time = time.time()  # Tiempo de inicio para esta imagen
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()  # Ejecutar la inferencia en TensorFlow Lite
    inference_time = time.time() - start_time
    total_inference_time += inference_time

    # Obtener predicción
    preds = interpreter.get_tensor(output_details[0]['index'])
    top_pred_idx = np.argmax(preds, axis=-1)

    print(f"Predicción: {cifar10_classes[top_pred_idx[0]]}, Tiempo: {inference_time:.4f}s")
    print(f"Etiqueta real: {cifar10_classes[label]}")

    # Contar los aciertos
    if top_pred_idx[0] == label:
        correct_predictions += 1
        print("✅ Acierto")

# Calcular el porcentaje de aciertos
accuracy = (correct_predictions / total_images) * 100

# Mostrar resultados
print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")
