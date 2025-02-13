# basat en el test de PC ajustant les rutes

import torch
import numpy as np
import time
from fastai.vision.all import *
from pathlib import Path

def main():
    
    # Importar 'accuracy' correctamente
    from fastai.metrics import accuracy
    # Ruta donde guardaste el modelo
    model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-ResNet_cifar10/modelsPytorch/stage-2-pytorch')
    
    # Cargar el modelo usando torch.load()
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Asegurarse de que el DataLoader esté bien configurado
    path = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images')
    dls = ImageDataLoaders.from_folder(path, valid='test', valid_pct=0.2, bs=1, item_tfms=Resize(32), batch_tfms=Normalize.from_stats(*imagenet_stats))
    
    # Re-crea el learner y asigna el modelo cargado
    learner = vision_learner(dls, resnet18, metrics=accuracy)
    learner.model = model  # Asignamos el modelo cargado a la red
    
    # Definir las clases de CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Preparar el conjunto de test
    test_dataset = dls.valid

    # Inicializar variables para métricas
    correct_predictions = 0
    total_images = 10
    total_inference_time = 0

    # Usar el DataLoader de validación para iterar sobre las imágenes
    for i, (image, label) in enumerate(dls.valid):
        if i >= total_images:
            break

        start_time = time.time()  # Tiempo de inicio para esta imagen

        # Asegúrate de que 'image' tenga las dimensiones correctas
        #image = image.permute(0, 3, 1, 2)  # Cambiar dimensiones a [batch_size, channels, height, width]
        #image = image.to(torch.float32)  # Convertir a tipo flotante
        #image = image / 255.0  # Normalizar los valores de los píxeles
        #image = image[0].unsqueeze(0)

        # Realizar la predicción
        with torch.no_grad():  # Desactiva el cálculo del gradiente
            prediction = learner.model(image)
        #print(prediction.shape)    
        predicted_class = prediction.argmax(dim=1).item()  # Obtener la clase con mayor probabilidad

        true_class = label.item()  # Clase verdadera
        print("Se predice: " + cifar10_classes[predicted_class])
        print("Resulta que es un: " + cifar10_classes[true_class])

        # Contar los aciertos
        if predicted_class == true_class:
            correct_predictions += 1
            print("✅")

        end_time = time.time()  # Tiempo de fin para esta imagen
        total_inference_time += (end_time - start_time)

    # Calcular el porcentaje de aciertos
    accuracyResult = (correct_predictions / total_images) * 100

    # Mostrar resultados
    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracyResult:.2f}%")

# Asegurarse de ejecutar la función solo cuando el script se ejecute como principal
if __name__ == '__main__':
    main()
