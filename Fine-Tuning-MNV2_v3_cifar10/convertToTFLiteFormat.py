import tensorflow as tf

# Cargar el modelo MobileNetV2 preentrenado con pesos de ImageNet
model = tf.keras.models.load_model(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\modelosFTuneados\mnv2_cifar10_bo_v1.h5')


# Configurar el convertidor de TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Activar optimizaciones
converter.target_spec.supported_types = [tf.float16]  # Reducir precisión a float16 (opcional, para reducir tamaño y mejorar rendimiento)

# Convertir el modelo a formato TensorFlow Lite
try:
    tflite_model = converter.convert()
    # Guardar el modelo convertido
    with open("mnv2_cifar10_bo_optimized.tflite", "wb") as f:
        f.write(tflite_model)
    print("Conversión a TensorFlow Lite exitosa.")
except Exception as e:
    print("Error en la conversión:", e)