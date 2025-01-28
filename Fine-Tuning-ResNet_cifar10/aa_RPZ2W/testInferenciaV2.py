# Basat en el V1 però sense processar tot el dataset de colp: processa images 1 a 1.

import torch
import numpy as np
import time
from fastai.vision.all import *
from pathlib import Path
import os
from PIL import Image
import random


def preprocess_image(image_path):
    # Cargar imagen y aplicar preprocesamiento mínimo
    img = Image.open(image_path) #.convert('RGB')
    #img = img.resize((32, 32))   # Redimensionar
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0)  # Cambiar dimensiones
    img_tensor = img_tensor.float() / 255.0  # Normalizar valores entre 0 y 1
    return img_tensor


def main():
    # Ruta del modelo
    model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-ResNet_cifar10/modelsPytorch/stage-2-pytorch')
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Definir las clases de CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

    # Ruta de imágenes
    image_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test')

    # Inicializar variables para métricas
    correct_predictions = 0
    total_inference_time = 0
    total_images = 3000

    # Procesar imágenes secuencialmente
    image_paths = list(Path(image_dir).glob('*/*.jpg'))[:total_images]
    random.shuffle(image_paths)
    image_paths = image_paths[:total_images]


    for i, image_path in enumerate(image_paths):
        image = preprocess_image(image_path)
        label = cifar10_classes.index(image_path.parent.name)

        start_time = time.time()
        
        # Realizar la predicción
        with torch.no_grad():
            prediction = model(image)
        
        predicted_class = prediction.argmax(dim=1).item()
        end_time = time.time()

        print(f"Se predice: {cifar10_classes[predicted_class]}")
        print(f"Resulta que es un: {cifar10_classes[label]}")

        # Contar los aciertos
        if predicted_class == label:
            correct_predictions += 1
            print("✅")

        total_inference_time += (end_time - start_time)

    # Calcular el porcentaje de aciertos
    accuracy_result = (correct_predictions / total_images) * 100

    # Mostrar resultados
    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy_result:.2f}%")

if __name__ == '__main__':
    main()
