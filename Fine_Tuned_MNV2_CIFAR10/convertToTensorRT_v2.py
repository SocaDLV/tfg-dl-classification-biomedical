import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

saved_model_dir = os.path.expanduser('~/codi/.../NEW_mnv2_cifar10_bo_v1_saved_model_format')

output_model_dir = os.path.expanduser('~/codi/.../modelosFTuneados')

params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params = params._replace(precision_mode="FP16")

converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=params)

converter.convert()

converter.save(output_model_dir)

print(f'Modelo convertido y guardado en {output_model_dir}')