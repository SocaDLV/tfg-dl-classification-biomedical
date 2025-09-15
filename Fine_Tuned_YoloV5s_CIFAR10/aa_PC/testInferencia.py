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

    model_path = Path(r'C:\Users\...\best16-1-25.pt')  

    model = torch.hub.load('ultralytics/yolov5' , 'custom' , path=str(model_path), force_reload=True) 
    model.eval() 

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    images_path = Path(r'C:\Users\...\content\images\test')

    transform = T.Compose([
        T.Resize((224, 224)), 
        T.ToTensor(),  
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])

    correct_predictions = 0
    total_images = 3000
    total_inference_time = 0

    image_files = list(images_path.rglob('*.jpg'))[:total_images]

    for i, img_path in enumerate(image_files):
        img = Image.open(img_path)                
        img_tensor = transform(img).unsqueeze(0)  

        start_time = time.time()
        with torch.no_grad():  
            outputs = model(img_tensor)  
        end_time = time.time()
        print(end_time - start_time)

        predictions = outputs[0]  
        predicted_class = predictions.argmax().item()  

        true_class = cifar10_classes.index(img_path.parent.name)

        if predicted_class == true_class:
            correct_predictions += 1

        total_inference_time += (end_time - start_time)

        print(f"Imagen {i + 1}/{total_images}")
        print(f"Predicción: {cifar10_classes[predicted_class]}, Real: {cifar10_classes[true_class]}")
        if predicted_class == true_class:
            print("✅ Correcto")
        else:
            print("❌ Incorrecto")

    accuracy_result = (correct_predictions / total_images) * 100

    print(f"\nTiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy_result:.2f}%")

    pathlib.PosixPath = temp

if __name__ == '__main__':
    main()