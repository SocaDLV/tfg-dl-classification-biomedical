import os

import tensorflow as tf
import numpy as np
import time
import tensorflow_datasets as tfds
from tensorflow.python.client import device_lib

def main():

  tf.config.optimizer.set_jit(True)  

  cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                    'dog', 'frog', 'horse', 'ship', 'truck']

  data_dir = os.path.expanduser('~/codi/.../content/images/')

  builder = tfds.folder_dataset.ImageFolder(data_dir)
  raw_test = builder.as_dataset(split='test', shuffle_files=True)

  IMG_SIZE = 224 

  def format_example(pair):
    image, label = pair['image'], pair['label']
    image = tf.cast(image, tf.float32)
    image = (image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

  test_dataset = raw_test.map(format_example).batch(1).prefetch(tf.data.AUTOTUNE)

  model_path = os.path.expanduser('~/codi/.../mnv2_cifar10_bo_v1_trt.bin')
  model = tf.keras.models.load_model(model_path, compile=False, custom_objects={})

  correct_predictions = 0
  total_images = 3000
  total_inference_time = 0

  for i, (image, label) in enumerate(test_dataset.take(total_images)):
      start_time = time.time()                  

      prediction = model.predict(image)         
      predicted_class = np.argmax(prediction)   
      print("Se predice "+str(predicted_class))
      true_class = label.numpy()               
      print("Resulta que es un "+str(true_class))

      if predicted_class == true_class:
          correct_predictions += 1
          print("✅")

      end_time = time.time()                    
      total_inference_time += (end_time - start_time)

  accuracy = (correct_predictions / total_images) * 100

  print(f"Tiempo total de inferencia: {total_inference_time:.2f} segundos")
  print(f"Imágenes correctamente clasificadas: {correct_predictions}/{total_images}")
  print(f"Porcentaje de aciertos: {accuracy:.2f}%")

if __name__ == '__main__':
    main()
