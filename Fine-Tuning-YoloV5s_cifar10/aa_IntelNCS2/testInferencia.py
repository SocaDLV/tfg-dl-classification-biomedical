# Basat en el test2 de ResNetCifar10 pa IntelNCS2, en detalls del YoloV5Cifar10 pa RPZ2W

# en device_name="CPU", es prou lento (0.5s per inferencia) i relativament precís (aprox. 75%?)
# en device_name="MYRIAD", es ràid (0.05s per inferencia) i precís (80%)

import os
import time
import numpy as np
import psutil
import cv2
import torch
import openvino.runtime as ov
from PIL import Image
from pathlib import Path
from random import sample
from torchvision import transforms

def main():

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Inicializar el Core de OpenVINO
    core = ov.Core()

    # Ruta al modelo ONNX
    model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-YoloV5s_cifar10/modelONNX/best16-1-25.onnx')
    
    # Cargar modelo
    model = core.read_model(model_path)

    # Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
    compiled_model = core.compile_model(model=model, device_name="MYRIAD")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

    # Definir las clases de CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    # Ruta a las imágenes de test
    data_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test')

    # Tamaño de imagen esperado por YoloV5s -> No ho utilitzem, les imatges ja estan a 32x32
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
        
        return sampled_files

    # Cargar las primeras 3000 imágenes
    total_images = 3000
    image_files = load_images(data_dir, total_images)

    # Inicializar variables para métricas
    correct_predictions = 0
    total_inference_time = 0

    # Necesari pal model
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    network_input_shape = input_key.shape

    # Realizar inferencia sobre las imágenes en batches de tamaño 1
    for i, file in enumerate(image_files):
        # Leer la imagen y obtener la etiqueta
        label = cifar10_classes.index(file.parent.name)  # Usar el nombre del subdirectorio como etiqueta

        # Procesar imagen
        image = Image.open(str(file))
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        start_time = time.time()  # Tiempo de inicio para esta imagen

        result = compiled_model(image_tensor)[output_key] #<-Comment o  Uncomment si algo falla

        inference_time = time.time() - start_time
        total_inference_time += inference_time

        #predicted_class = result.argmax(dim=1).item()

        # Obtener predicción
        result_index = np.argmax(result)
        print(f"Predicción para imagen {i}: {cifar10_classes[result_index]}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
        print(f"Etiqueta real: {cifar10_classes[label]}")

        # Contar los aciertos
        if result_index == label:
            correct_predictions += 1
            print("✅ Acierto")

    # Calcular el porcentaje de aciertos
    accuracy = (correct_predictions / total_images) * 100

    # Mostrar resultados
    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
