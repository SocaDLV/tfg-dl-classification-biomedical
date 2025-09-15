import os
import re
import time
import numpy as np
import psutil
import cv2
import tflite_runtime.interpreter as tflite
from pathlib import Path

model_path = os.path.expanduser('~/codi/.../mobilenet_v2_optimized.tflite')
interpreter = tflite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        interpreter.set_tensor(input_details[0]['index'], img)
        start_time = time.perf_counter()
        interpreter.invoke()  
        inference_time = time.perf_counter() - start_time
        total_inference_time += inference_time

        preds = interpreter.get_tensor(output_details[0]['index'])
        top_pred_idx = np.argmax(preds, axis=-1)
        print(f"Predicción para {img_file}: {top_pred_idx}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    
    cpu_usage_end = psutil.cpu_percent(interval=1)
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

image_folder = Path(os.path.expanduser('~/codi/.../tinyImageNet3K'))
classify_and_measure(image_folder)