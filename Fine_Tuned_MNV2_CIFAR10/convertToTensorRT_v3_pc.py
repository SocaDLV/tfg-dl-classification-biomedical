import os
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

path_to_savedmodel = r'C:\Users\...\NEW_mnv2_cifar10_bo_v1_saved_model_format'

converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path_to_savedmodel
        )

converter.convert()
print("Conversión con éxito!")

converter.save("mnv2_cifar10_bo_trt")
print("Guardado con éxito!")
