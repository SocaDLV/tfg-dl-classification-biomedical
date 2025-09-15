import cv2
import torch
import numpy as np
from torchvision import transforms
from fastai.vision.all import *
import time

def main():
    # Ruta al modelo
    model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsPytorch\stage-2-pytorch'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()

    # Clases de CIFAR-10
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    # Transformación de imagen como en entrenamiento
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Iniciar webcam
    cap = cv2.VideoCapture(0)  # Usa 0 o 1 según tu sistema

    # Cambiar resolución a 320x240
    cap.set(3, 320)
    cap.set(4, 240)

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

            # Procesar frame (solo centro si quieres)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mostrar resolución original capturada
            print(f"Resolución capturada: {img.shape[1]}x{img.shape[0]}")

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

            # Mostrar frame en ventana (opcional)
            cv2.imshow('Webcam (presiona q para salir)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()