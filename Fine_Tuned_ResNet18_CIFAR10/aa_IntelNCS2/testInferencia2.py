import os
import time
import numpy as np
import psutil
import cv2
import openvino.runtime as ov
from PIL import Image
from pathlib import Path
from random import sample
from torchvision import transforms

def main():

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    core = ov.Core()

    model_path = os.path.expanduser('~/codi/.../stage-2.onnx')
    model = core.read_model(model_path)

    compiled_model = core.compile_model(model=model, device_name="MYRIAD")  

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

    data_dir = os.path.expanduser('~/codi/.../content/images/test')

    IMG_SIZE = 32

    def load_images(data_dir, total_images):
        data_dir = Path(data_dir).expanduser()
        image_files = list(data_dir.rglob('*.jpg'))  
        
        if len(image_files) == 0:
            raise ValueError("No se encontraron imágenes en el directorio especificado.")
        
        total_images = min(len(image_files), total_images)
        sampled_files = sample(image_files, total_images)
        
        return sampled_files

    total_images = 3000
    image_files = load_images(data_dir, total_images)

    correct_predictions = 0
    total_inference_time = 0

    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    network_input_shape = input_key.shape

    for i, file in enumerate(image_files):
        label = cifar10_classes.index(file.parent.name)  

        image = Image.open(str(file))
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        start_time = time.time()  
        result = compiled_model(image_tensor)[output_key] 

        inference_time = time.time() - start_time
        total_inference_time += inference_time

        result_index = np.argmax(result)
        print(f"Predicción para imagen {i}: {cifar10_classes[result_index]}")
        print(f"Tiempo de inferencia: {inference_time:.4f} segundos")
        print(f"Etiqueta real: {cifar10_classes[label]}")

        if result_index == label:
            correct_predictions += 1
            print("✅ Acierto")

    accuracy = (correct_predictions / total_images) * 100

    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
