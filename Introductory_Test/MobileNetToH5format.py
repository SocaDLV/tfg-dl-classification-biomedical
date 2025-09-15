from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet')

model.save("mobilenetv2.h5")