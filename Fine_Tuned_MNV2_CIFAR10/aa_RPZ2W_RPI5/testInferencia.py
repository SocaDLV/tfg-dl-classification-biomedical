import os
import re
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
import tensorflow_datasets as tfds
from pathlib import Path

model_path = os.path.expanduser('~/codi/.../mnv2_cifar10_bo_optimized.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

data_dir = '~codi\...\content\images'

builder = tfds.folder_dataset.ImageFolder(data_dir)
raw_test = builder.as_dataset(split='test', shuffle_files=True)

IMG_SIZE = 224 

def format_example(pair):
  image, label = pair['image'], pair['label']
  image = (image/127.5) - 1
  image = cv2.resize(image, (224, 224))
  return image, label

test_dataset = raw_test.map(format_example).batch(1) 

correct_predictions = 0
total_images = 3000
total_inference_time = 0

for i, (image, label) in enumerate(test_dataset.take(total_images)):
    start_time = time.time()                  

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()                      
    inference_time = time.time() - start_time
    total_inference_time += inference_time

    preds = interpreter.get_tensor(output_details[0]['index'])
    top_pred_idx = np.argmax(preds, axis=-1)

    print(f"Predicción para {image}: {top_pred_idx}")
    print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

    true_class = label.numpy()

    print("Resulta que es un "+str(true_class))  
    print(int((top_pred_idx)[0]))                

    if top_pred_idx == true_class:
        correct_predictions += 1
        print("✅")

    end_time = time.time()                       
    total_inference_time += (end_time - start_time)

accuracy = (correct_predictions / total_images) * 100

print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")
