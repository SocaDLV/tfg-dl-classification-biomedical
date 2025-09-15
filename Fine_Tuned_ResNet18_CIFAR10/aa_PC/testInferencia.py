from fastai.vision.all import *
import time
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random

cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

path = Path(r'C:\Users\...\content\images')

dls = ImageDataLoaders.from_folder(str(path), valid='test')

learner = cnn_learner(dls, resnet18, metrics=accuracy)
learner.load(str(Path(r'C:\Users\...\stage-2-pytorch')),cpu=False)

correct_predictions = 0
total_images = 3000
total_inference_time = 0

IMG_SIZE = 32  

def format_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img


test_images = []
for class_name in cifar10_classes:
    class_folder = data_dir / class_name
    all_images = list(class_folder.glob('*.jpg'))  
    selected_images = random.sample(all_images, 300)  
    test_images.extend(selected_images)

for i, image_path in enumerate(test_images):
    image = format_image(image_path)
    start_time = time.time()  

    pred_class, pred_idx, outputs = learner.predict(image)
    predicted_class = pred_class
    true_class = image_path.parent.name  

    print(f"Se predice {predicted_class}")
    print(f"Resulta que es un {true_class}")

    if predicted_class == true_class:
        correct_predictions += 1
        print("✅")

    end_time = time.time() 
    total_inference_time += (end_time - start_time)

accuracy = (correct_predictions / total_images) * 100

print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")