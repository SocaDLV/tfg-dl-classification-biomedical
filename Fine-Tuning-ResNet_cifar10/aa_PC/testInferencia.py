from fastai.vision.all import *
import time
import numpy as np
import os
from pathlib import Path
from PIL import Image
import random

# Definir las clases de CIFAR-10
cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

path = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\aa_PC\content\images')

# Cargar DataLoaders
dls = ImageDataLoaders.from_folder(str(path), valid='test')

# Ahora cargar el Learner
learner = cnn_learner(dls, resnet18, metrics=accuracy)
learner.load(str(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsPytorch\stage-2-pytorch')),cpu=False)

# Inicializar variables para métricas
correct_predictions = 0
total_images = 3000
total_inference_time = 0

# Preparar el conjunto de test
IMG_SIZE = 32  # Tamaño que el modelo espera

def format_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img

# Seleccionar 300 imágenes aleatorias de cada clase para obtener un total de 3000 imágenes
test_images = []
for class_name in cifar10_classes:
    class_folder = data_dir / class_name
    all_images = list(class_folder.glob('*.jpg'))  # Suponiendo que las imágenes son .jpg
    selected_images = random.sample(all_images, 300)  # Seleccionar 300 imágenes aleatorias de esta clase
    test_images.extend(selected_images)

# Realizar inferencia sobre 3000 imágenes
for i, image_path in enumerate(test_images):
    image = format_image(image_path)
    start_time = time.time()  # Tiempo de inicio para esta imagen

    # Realizar la predicción
    pred_class, pred_idx, outputs = learner.predict(image)
    predicted_class = pred_class
    true_class = image_path.parent.name  # El nombre de la carpeta es la clase verdadera

    print(f"Se predice {predicted_class}")
    print(f"Resulta que es un {true_class}")

    # Contar los aciertos
    if predicted_class == true_class:
        correct_predictions += 1
        print("✅")

    end_time = time.time()  # Tiempo de fin para esta imagen
    total_inference_time += (end_time - start_time)

# Calcular el porcentaje de aciertos
accuracy = (correct_predictions / total_images) * 100

# Mostrar resultados
print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
print(f"Porcentaje de aciertos: {accuracy:.2f}%")