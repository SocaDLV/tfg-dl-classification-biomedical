import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Ruta del modelo SavedModel
saved_model_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/NEW_mnv2_cifar10_bo_v1_saved_model_format')

# Ruta de salida para el modelo convertido
output_model_dir = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados')

# Configurar los parámetros de conversión
params = trt.DEFAULT_TRT_CONVERSION_PARAMS
params = params._replace(precision_mode="FP16")#, maximum_cached_engine_count=100)

# Crear el convertidor de TensorRT
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=params)

# Realizar la conversión a TensorRT
converter.convert()

# Guardar el modelo optimizado en TensorRT
converter.save(output_model_dir)

print(f'Modelo convertido y guardado en {output_model_dir}')

