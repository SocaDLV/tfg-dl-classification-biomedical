import torch
import numpy as np
import time
from fastai.vision.all import *
from pathlib import Path

def main():
    
    from fastai.metrics import accuracy
    model_path = r'C:\Users\...\stage-2-pytorch'
    
    model = torch.load(model_path, map_location=torch.device('cpu'))
    
    path = Path(r'C:\Users\...\content\images')
    dls = ImageDataLoaders.from_folder(path, valid='test', valid_pct=0.2, bs=1, batch_tfms=Normalize.from_stats(*imagenet_stats))
    
    learner = vision_learner(dls, resnet18, metrics=accuracy)
    learner.model = model  
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    test_dataset = dls.valid

    correct_predictions = 0
    total_images = 3000
    total_inference_time = 0

    for i, (image, label) in enumerate(dls.valid):
        if i >= total_images:
            break

        start_time = time.time()  

        
        with torch.no_grad():  
            prediction = learner.model(image)
        predicted_class = prediction.argmax(dim=1).item()  

        true_class = label.item()  
        print("Se predice: " + cifar10_classes[predicted_class])
        print("Resulta que es un: " + cifar10_classes[true_class])

        if predicted_class == true_class:
            correct_predictions += 1
            print("✅")

        end_time = time.time()  
        total_inference_time += (end_time - start_time)

    accuracyResult = (correct_predictions / total_images) * 100

    print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracyResult:.2f}%")

if __name__ == '__main__':
    main()