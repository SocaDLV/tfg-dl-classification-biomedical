import os
import re
import time
import numpy as np
import psutil

import cv2
import numpy as np
import openvino.runtime as ov

from pathlib import Path
from PIL import Image

core = ov.Core()

model = core.read_model(r"C:\Users\...\model_ir\saved_model.xml")

compiled_model = core.compile_model(model=model, device_name="MYRIAD")

def load_labels(labels_path):
    labels = {}
    with open(labels_path, 'r') as f:
        for line in f:
            img_name, label = line.strip().split()[:2]
            labels[img_name] = label
    return labels

labels_dict = load_labels(Path(r'C:\Users\...\IntelNCS2_RPZ2W_validation_imgs_correct_preds.txt'))

image_folder = Path(r'C:\Users\...\tinyImageNet3K_validation')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  
    img = img.astype(np.float32)       
    img = img / 255.0                  
    img = img[np.newaxis, :]           
    return img

input_key = compiled_model.input(0)
output_key = compiled_model.output(0)

network_input_shape = input_key.shape

def natural_sort_key(filename):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', filename)]

def classify_and_measure(image_folder):
    image_files = [os.path.join(image_folder, f) for f in sorted(os.listdir(image_folder), key=natural_sort_key) if f.endswith(('.jpg', '.jpeg', '.png', '.JPEG'))]
    
    total_inference_time = 0
    cpu_usage_start = psutil.cpu_percent(interval=1)       

    correct_predictions = 0
    total_images = 0

    for img_file in image_files:
        img = preprocess_image(img_file)
        
        start_time = time.perf_counter()                   
        result = compiled_model(img)[output_key]           
        inference_time = time.perf_counter() - start_time  
        total_inference_time += inference_time

        result_index = np.argmax(result)
        print(f"Predicción para {img_file}: {result_index}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

        true_label = labels_dict[os.path.basename(img_file)]  
        
        if int(result_index) == int(true_label):
            correct_predictions += 1
            print(f"La predicción es correcta! ✅")
        total_images += 1
    

    cpu_usage_end = psutil.cpu_percent(interval=1)         
    
    print(f"Tiempo total de inferencia: {total_inference_time:.4f} segundos")
    print(f"Uso medio del CPU: {((cpu_usage_start + cpu_usage_end) / 2):.2f}%")

    print(f"Se han acertado {correct_predictions} del total de {total_images}")
    accuracy = correct_predictions / total_images
    print(f"Precisión en las {total_images} imágenes: {accuracy * 100:.2f}%")

classify_and_measure(image_folder)