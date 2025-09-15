import os
import re
import time
import numpy as np
import psutil

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions

from pathlib import Path
from PIL import Image

model = MobileNetV2(weights='imagenet')

image_folder = Path(r'C:\Users\...\tinyImageNet3K')

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224,224))
    
    img_array = image.img_to_array(img)
    
    if img_array.shape[-1] == 1:
        img_array = np.repeat(img_array, 3, axis=-1)
    
    img_array = preprocess_input(img_array)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=None)  

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.time()
        preds = model.predict(img) 
        inference_time = time.time() - start_time  
        total_inference_time += inference_time

        decoded_preds = decode_predictions(preds, top=3)[0]
        print(f"Predicción para {img_file}: {decoded_preds}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
    
    cpu_usage_end = psutil.cpu_percent(interval=None)  
    
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

classify_and_measure(image_folder)