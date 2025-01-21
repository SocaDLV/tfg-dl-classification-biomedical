import torch
import torchvision.transforms as T
import numpy as np
import time
import pathlib
from pathlib import Path
from PIL import Image

def main():

    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath

    # Ruta donde guardaste el modelo
    model_path = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-YoloV5s_cifar10\model\best16-1-25.pt')  # El que vaig guardar originalment
    # model_path = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-YoloV5s_cifar10\yolov5\weigths\yolov5s-cls.pt')  # Uno random


    # Cargar el modelo
    model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True) #, map_location=torch.device('cpu'))
    model.eval()  # Configurar el modelo en modo evaluación

    # Definir las clases de CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # Ruta de las imágenes
    images_path = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\aa_PC\content\images\test')

    # Transformaciones para las imágenes (escalado, normalización, etc.)
    transform = T.Compose([
        T.Resize((224, 224)),  # Escalar a tamaño esperado por YOLOv5 -> 224x224
        T.ToTensor(),  # Convertir a tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalización estándar
    ])

    # Inicializar variables para métricas
    correct_predictions = 0
    total_images = 3000
    total_inference_time = 0

    # Iterar sobre las imágenes del conjunto de prueba
    image_files = list(images_path.rglob('*.jpg'))[:total_images]

    for i, img_path in enumerate(image_files):
        # Cargar y preprocesar la imagen
        img = Image.open(img_path)                #.convert("RGB") -> No afecta en res al resultat
        img_tensor = transform(img).unsqueeze(0)  # Añadir dimensión de batch

        # Realizar la predicción
        start_time = time.time()
        with torch.no_grad():  # Desactivar el cálculo del gradiente
            outputs = model(img_tensor)  # Inferencia
        end_time = time.time()
        print(end_time - start_time)

        # Procesar las predicciones
        predictions = outputs[0]  # Salida del modelo
        predicted_class = predictions.argmax().item()  # Clase con mayor probabilidad

        # Extraer la clase verdadera desde el nombre del archivo o la etiqueta
        true_class = cifar10_classes.index(img_path.parent.name)

        # Contar los aciertos
        if predicted_class == true_class:
            correct_predictions += 1

        # Acumular el tiempo de inferencia
        total_inference_time += (end_time - start_time)

        # Imprimir resultados parciales
        print(f"Imagen {i + 1}/{total_images}")
        print(f"Predicción: {cifar10_classes[predicted_class]}, Real: {cifar10_classes[true_class]}")
        if predicted_class == true_class:
            print("✅ Correcto")
        else:
            print("❌ Incorrecto")

    # Calcular el porcentaje de aciertos
    accuracy_result = (correct_predictions / total_images) * 100

    # Mostrar resultados finales
    print(f"\nTiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy_result:.2f}%")

    pathlib.PosixPath = temp

# Asegurarse de ejecutar la función solo cuando el script se ejecute como principal
if __name__ == '__main__':
    main()
