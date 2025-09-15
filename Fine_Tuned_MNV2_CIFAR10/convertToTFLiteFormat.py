import tensorflow as tf

model = tf.keras.models.load_model(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\modelosFTuneados\mnv2_cifar10_bo_v1.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  
converter.target_spec.supported_types = [tf.float16]  

try:
    tflite_model = converter.convert()
    with open("mnv2_cifar10_bo_optimized.tflite", "wb") as f:
        f.write(tflite_model)
    print("Conversión a TensorFlow Lite exitosa.")

except Exception as e:
    print("Error en la conversión:", e)