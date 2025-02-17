import tensorflow as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt

path_to_savedmodel = os.path.expanduser('~/codi/TFG/Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/NEW_mnv2_cifar10_bo_v1_saved_model_format')
converter = trt.TrtGraphConverterV2(input_saved_model_dir=path_to_savedmodel, precision_mode=trt.TrtPrecisionMode.FP16)
converter.convert()
converter.save("mnv2_cifar10_bo_trt")