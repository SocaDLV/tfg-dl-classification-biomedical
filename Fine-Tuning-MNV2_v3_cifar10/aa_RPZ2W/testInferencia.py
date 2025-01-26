#Basat en el test de inferencia de PC de MNV2_v3_cifar10

import os
import re
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
import tensorflow_datasets as tfds
from pathlib import Path

# Cargar el modelo TensorFlow Lite en la Raspberry Pi
model_path = '~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/mnv2_cifar10_bo_optimized.tflite'
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Obtener detalles de entrada y salida del modelo
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Definir las clases de CIFAR-10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

# Ruta a las imágenes de test
data_dir = '~codi\Fine-Tuning-MNV2_v3_cifar10\aa_PC\content\images'

# Cargar las imágenes desde la carpeta local
builder = tfds.folder_dataset.ImageFolder(data_dir)
raw_test = builder.as_dataset(split='test', shuffle_files=True)

IMG_SIZE = 224 # All images will be resized to 224x224 -> for MobileNetV2

def format_example(pair):
  image, label = pair['image'], pair['label']
  #image = tf.cast(image, tf.float32) -> No compatible en TF
  image = (image/127.5) - 1
  image = cv2.resize(image, (224, 224))
  return image, label

# Preparar el conjunto de test
test_dataset = raw_test.map(format_example).batch(1) #.prefetch(tf.data.AUTOTUNE) -> No compatible en TF

# Inicializar variables para métricas
correct_predictions = 0
total_images = 3000
total_inference_time = 0

# Realizar inferencia sobre 3000 imágenes
for i, (image, label) in enumerate(test_dataset.take(total_images)):
    start_time = time.time()                  # Tiempo de inicio para esta imagen

    # Realizar la predicción

    #image = tf.expand_dims(image, axis=0)    # Añadir la dimensión de batch, i no compatible en TF

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()                      # Ejecutar la inferencia en TensorFlow Lite
    inference_time = time.time() - start_time
    total_inference_time += inference_time

    # Obtener predicción y ver si es correcta
    preds = interpreter.get_tensor(output_details[0]['index'])
    top_pred_idx = np.argmax(preds, axis=-1)
    print(f"Predicción para {image}: {top_pred_idx}")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    true_class = label.numpy()
    print("Resulta que es un "+str(true_class))  # Comprobar que imprime bien la etiqueta real
    print(int((top_pred_idx)[0]))                # Comprobar que imprime bien la etiqueta inferida


    # Contar los aciertos
    if top_pred_idx == true_class:
        correct_predictions += 1
        print("✅")

    end_time = time.time()                       # Tiempo de fin para esta imagen
    total_inference_time += (end_time - start_time)

# Calcular el porcentaje de aciertos
accuracy = (correct_predictions / total_images) * 100

# Mostrar resultados
print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")