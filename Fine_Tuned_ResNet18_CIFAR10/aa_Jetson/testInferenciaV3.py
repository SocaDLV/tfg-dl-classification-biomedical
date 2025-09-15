import os
import time
import numpy as np
from pathlib import Path
import onnxruntime as ort
from PIL import Image
import random
from torchvision import transforms

transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def preprocess_image(image_path, img_size=32):
    img_array = Image.open(image_path).convert('RGB')
    img_array = transform(img_array)
    img_array = img_array.unsqueeze(0)

    return img_array.numpy()  

def main():
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']

    providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
    
    onnx_model_path = os.path.expanduser('~/codi/.../stage-2-Jetson.onnx')
    session = ort.InferenceSession(onnx_model_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    data_dir = os.path.expanduser('~/codi/.../content/images/test')
    data_path = Path(data_dir)
    
    all_image_paths = list(data_path.rglob('*.jpg'))
    if not all_image_paths:
        raise ValueError("No se encontraron imágenes en el directorio especificado.")

    random.shuffle(all_image_paths)
    
    total_images = 3000
    selected_paths = all_image_paths[:total_images]

    correct_predictions = 0
    total_inference_time = 0.0

    for i, image_path in enumerate(selected_paths):
        true_class_name = image_path.parent.name
        if true_class_name not in cifar10_classes:
            continue
        true_class = cifar10_classes.index(true_class_name)

        input_image = preprocess_image(image_path, img_size=32)

        start_time = time.time()
        outputs = session.run([output_name], {input_name: input_image})
        inference_time = time.time() - start_time
        total_inference_time += inference_time

        prediction = outputs[0]
        predicted_class = np.argmax(prediction, axis=1)[0]

        print(f"Imagen {i+1}/{total_images}:")
        print(f"  Predicción: {cifar10_classes[predicted_class]}")
        print(f"  Etiqueta real: {cifar10_classes[true_class]}")
        print(f"  Tiempo de inferencia: {inference_time:.4f} segundos")
        if predicted_class == true_class:
            correct_predictions += 1
            print("  ✅ Correcto")
        else:
            print("  ❌ Incorrecto")
        print("-----------------------------------------------------")

    accuracy = (correct_predictions / total_images) * 100

    print(f"\nTiempo total de inferencia: {total_inference_time:.2f} segundos")
    print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
    print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
