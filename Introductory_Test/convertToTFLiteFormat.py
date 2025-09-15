import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
converter.target_spec.supported_types = [tf.float16]  

try:
    tflite_model = converter.convert()
    with open("mobilenet_v2_optimized.tflite", "wb") as f:
        f.write(tflite_model)
    print("Conversión a TensorFlow Lite exitosa.")
except Exception as e:
    print("Error en la conversión:", e)
