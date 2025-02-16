# Basat en el testInferencia.py de PC, pero en canvis de rutes+detecció GPU Jetson

import os

# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' -> Vore si en aso comentat m'ix un warning de les Ops OneDNN

import tensorflow as tf
import numpy as np
import time
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

def main():

  tf.config.optimizer.set_jit(True)  # Habilitar XLA

  #Checks pa vore si detecta la GPU:
  #print("El nom de la GPU es: " + str(tf.test.gpu_device_name()))
  #print("La llista de dispositius es: " + str(device_lib.list_local_devices()))

  # Definir las clases de CIFAR-10
  cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

  # Ruta a las imágenes de test
  data_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/aa_PC/content/images/')

  # Cargar las imágenes desde la carpeta local
  builder = tfds.folder_dataset.ImageFolder(data_dir)
  raw_test = builder.as_dataset(split='test', shuffle_files=True)

  IMG_SIZE = 224 # All images will be resized to 224x224 -> for MobileNetV2

  def format_example(pair):
    image, label = pair['image'], pair['label']
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

  # Preparar el conjunto de test
  test_dataset = raw_test.map(format_example).batch(1).prefetch(tf.data.AUTOTUNE)

  # Cargar el modelo entrenado
  model_path = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/mnv2_cifar10_bo_v1.h5')
  model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})

  # Inicializar variables para métricas
  correct_predictions = 0
  total_images = 3000
  total_inference_time = 0

  # Realizar inferencia sobre 3000 imágenes
  for i, (image, label) in enumerate(test_dataset.take(total_images)):
      start_time = time.time()                  # Tiempo de inicio para esta imagen

      # Realizar la predicción
      #image = tf.expand_dims(image, axis=0)    # Añadir la dimensión de batch
      prediction = model.predict(image)         # Predicción para la imagen
      predicted_class = np.argmax(prediction)   # Clase con mayor probabilidad
      print("Se predice "+str(predicted_class))
      true_class = label.numpy()                # Clase verdadera
      print("Resulta que es un "+str(true_class))

      # Contar los aciertos
      if predicted_class == true_class:
          correct_predictions += 1
          print("✅")

      end_time = time.time()                    # Tiempo de fin para esta imagen
      total_inference_time += (end_time - start_time)

  # Calcular el porcentaje de aciertos
  accuracy = (correct_predictions / total_images) * 100

  # Mostrar resultados
  print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
  print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
  print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
