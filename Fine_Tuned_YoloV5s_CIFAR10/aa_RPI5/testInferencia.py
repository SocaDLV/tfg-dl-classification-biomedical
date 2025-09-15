import os
import torch
import numpy as np
import time
from PIL import Image
from pathlib import Path
from random import sample
from torchvision import transforms

def main():
    
    transform = transforms.Compose([
        transforms.Resize((128,128)),   
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model_path = os.path.expanduser('~/codi/.../best16-1-25')

    model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True) 
    model.eval()  
        
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

    path = Path(os.path.expanduser('~/codi/.../content/images/test'))
    total_images = 3000

    images = []
    for class_dir in path.iterdir():
        if class_dir.is_dir():
            all_images = list(class_dir.rglob("*.jpg"))
            sampled_images = sample(all_images, min(len(all_images), total_images // len(cifar10_classes)))
            images.extend(sampled_images)

    correct_predictions = 0
    total_inference_time = 0

    for image_path in images:
        label_str = image_path.parent.name
        true_class = cifar10_classes.index(label_str)

        image = Image.open(image_path)
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        start_time = time.time()
        with torch.no_grad():
            prediction = model(image_tensor)
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        predicted_class = prediction.argmax(dim=1).item()

        print(f"Predicción: {cifar10_classes[predicted_class]}, Tiempo: {inference_time:.4f}s")
        print(f"Etiqueta real: {cifar10_classes[true_class]}")

        if predicted_class == true_class:
            correct_predictions += 1
            print("✅")

    accuracy_result = (correct_predictions / len(images)) * 100
    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{len(images)}")
    print(f"Porcentaje de aciertos: {accuracy_result:.2f}%")

if __name__ == '__main__':
    main()