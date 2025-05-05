
# BASAT en el test RPZ2W V2 de Yolo

import os
import numpy as np
import time
import random
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from random import sample
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),  # OJO el resize estava mal, no es 224x224 sino 128x128
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path, img_size=32):
    img_array = Image.open(image_path).convert('RGB')
    img_array = transform(img_array)
    img_array = img_array.unsqueeze(0)

    return img_array.numpy()  # Devuelve en formato NHWC (1, 224, 224, 3)


def main():

    # Definir las clases CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']

    # Configurar ONNX Runtime para usar la GPU (si está disponible)
    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # Ruta al modelo ONNX (asegúrate de haber convertido tu modelo SavedModel a ONNX previamente)
    onnx_model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-YoloV5s_cifar10/modelONNX/best16-1-25-Jetson_fix.onnx')
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    # Obtener los nombres de entrada y salida del modelo
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Ruta a las imágenes de test (imágenes organizadas en subdirectorios con nombres de clase)
    data_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/test')
    data_path = Path(data_dir)

    # Recopilar todas las imágenes .jpg de los subdirectorios
    all_image_paths = list(data_path.rglob('*.jpg'))
    if not all_image_paths:
        raise ValueError("No se encontraron imágenes en el directorio especificado.")

    # Mezclar aleatoriamente para obtener una selección aleatoria
    random.shuffle(all_image_paths)
    
    # Seleccionar 3000 imágenes
    total_images = 3000
    selected_paths = all_image_paths[:total_images]

    # Variables para métricas
    correct_predictions = 0
    total_inference_time = 0.0

    for i, image_path in enumerate(selected_paths):
        # Obtener la etiqueta verdadera a partir del nombre del subdirectorio
        true_class_name = image_path.parent.name
        if true_class_name not in cifar10_classes:
            # Si la carpeta no corresponde a ninguna de las clases, se salta la imagen
            continue
        true_class = cifar10_classes.index(true_class_name)

        # Preprocesar la imagen
        input_image = preprocess_image(image_path, img_size=32)

        # Realizar la inferencia
        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_image})
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        # Asumimos que la salida es un vector de probabilidades (o logits ya pasados por softmax)
        prediction = outputs[0]
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Imprimir resultados parciales
        print(f"  Imagen {i+1}/{total_images}:")
        print(f"  Predicción: {cifar10_classes[predicted_class]}")
        print(f"  Etiqueta real: {cifar10_classes[true_class]}")
        print(f"  Tiempo de inferencia: {inference_time:.4f} segundos")
        if predicted_class == true_class:
            correct_predictions += 1
            print("  ✅ Correcto")
        else:
            print("  ❌ Incorrecto")
        print("-----------------------------------------------------")

    # Calcular el porcentaje de aciertos
    accuracy = (correct_predictions / total_images) * 100

    # Mostrar resultados finales
    print(f"\nTiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()