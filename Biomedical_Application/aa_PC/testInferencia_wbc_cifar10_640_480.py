import cv2
import torch
import numpy as np
from torchvision import transforms
from fastai.vision.all import *
import time

def main():
    model_path = r'C:\Users\...\stage-2-pytorch'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)  

    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam")
        return

    print("✅ Webcam iniciada. Pulsa 'q' para salir.")

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al capturar frame")
                break

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_resized = transform(img).unsqueeze(0)

            start = time.time()
            output = model(img_resized)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            confidence, predicted_class = torch.max(probabilities, 0)
            end = time.time()

            label = cifar10_classes[predicted_class.item()]
            confidence_pct = confidence.item() * 100

            if confidence_pct >= 70:
                print(f"DETECTADO: {label} ({confidence_pct:.1f}%)")
            else:
                print(f"Nada detectado: {label} ({confidence_pct:.1f}%)")

            cv2.imshow('Webcam (presiona q para salir)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()