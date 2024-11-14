from tensorflow.keras.applications import MobileNetV2

# Cargar el modelo preentrenado
model = MobileNetV2(weights='imagenet')

# Guardar el modelo en formato .h5
model.save("mobilenetv2.h5")