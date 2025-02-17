import os
import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

path_to_savedmodel = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/NEW_mnv2_cifar10_bo_v1_saved_model_format')

#params = trt.TrtConverterParams(precision_mode="FP16")
#converter = trt.TrtGraphConverterV2(input_saved_model_dir=path_to_savedmodel, conversion_params=params)

#converter = tf.experimental.tensorrt.Converter(
#        input_saved_model_dir=path_to_savedmodel
        #precision_mode='FP16'
#        )

converter = trt.TrtGraphConverterV2(
        input_saved_model_dir=path_to_savedmodel
        )

converter.convert()
print("Conversión con éxito!")
converter.save("mnv2_cifar10_bo_trt")
print("Guardado con éxito!")
