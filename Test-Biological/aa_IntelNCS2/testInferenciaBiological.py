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
        # Provar en i sense el normalize
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Inicializar el Core de OpenVINO
    core = ov.Core()

    # Cargar el modelo ONNX
    model_path = os.path.expanduser('~/codi/TFG/Test-Biological/models/biological-stage-2.onnx')
    model = core.read_model(model_path)

    # Compilar el modelo para el dispositivo especificado (CPU o MYRIAD)
    compiled_model = core.compile_model(model=model, device_name="MYRIAD")  # Cambiar "CPU" a "MYRIAD" cuando MYRIAD esté disponible

    # Definir las clases de CIFAR-10
    biological_classes = ['actinic_keratoses', 'chickenpox', 'cowpox', 'dermatofibroma', 'measles', 'melanoma']

    # --- IO del modelo ---
    input_key = compiled_model.input(0)
    output_key = compiled_model.output(0)
    n, c, h, w = input_key.shape  # NCHW esperado por el modelo

    # Iniciar webcam
    cap = cv2.VideoCapture(0)  # Usa 0 o 1 según tu sistema

    # Cambiar resolución a 320x240
    cap.set(3, 320)
    cap.set(4, 240)

    if not cap.isOpened():
        print("❌ No se pudo abrir la webcam")
        return

    print("✅ Webcam iniciada. Pulsa 'q' para salir.")
    printed_once = False
    THRESHOLD = 0.70

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("❌ Error al capturar frame")
                break

            if not printed_once:
                aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                print(f"Resolución capturada real: {aw}x{ah}")
                if (h, w) != (ah, aw):
                    print(f"⚠️ Aviso: el modelo espera {w}x{h} (W×H). "
                          "Si no coincide con la cámara, habrá que reescalar o reexportar el modelo.")
                printed_once = True

            # BGR -> RGB (sin resize)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # A tensor CHW y luego NCHW (tu transform ya lo hace)
            pil_img = Image.fromarray(rgb)   # 320x240
            tensor = transform(pil_img).unsqueeze(0)   # NCHW (torch tensor)
            inp = tensor.numpy()                       # OpenVINO usa numpy

            # Inferencia
            t0 = time.time()
            result = compiled_model(inp)[output_key]   # numpy
            infer_ms = (time.time() - t0) * 1000.0

            # Softmax + top-1
            logits = result[0].astype(np.float32)
            logits -= np.max(logits)
            probs = np.exp(logits) / np.sum(np.exp(logits))
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = biological_classes[idx] if idx < len(biological_classes) else f"class_{idx}"

            if conf >= THRESHOLD:
                print(f"DETECTADO: {label} ({conf*100:.1f}%) | {infer_ms:.1f} ms")
            else:
                print(f"Nada detectado: {label} ({conf*100:.1f}%) | {infer_ms:.1f} ms")

            # Mostrar (opcional)
            #cv2.imshow('Webcam (presiona q para salir)', frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
                #break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
