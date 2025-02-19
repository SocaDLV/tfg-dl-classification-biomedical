import tensorflow as tf
import tf2onnx
import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Cargar el modelo H5
model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\modelosFTuneados\mnv2_cifar10_bo_v1.h5'
model = tf.keras.models.load_model(model_path)

# Crear un concrete function con batch size = 1 para cada entrada
concrete_inputs = [
    tf.TensorSpec([1] + list(model_input.shape)[1:], model_input.dtype)
    for model_input in model.inputs
]
# Define la función concreta
full_model = tf.function(lambda inputs: model(inputs))
concrete_function = full_model.get_concrete_function(concrete_inputs)

# Imprimir nombres de entradas y salidas (para ver que se fija correctamente la forma)
input_names = [inp.name for inp in concrete_function.inputs]
output_names = [out.name for out in concrete_function.outputs]
print("Inputs:", input_names)
print("Outputs:", output_names)

# Convertir variables a constantes (congelar el grafo)
frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_graph_def = frozen_func.graph.as_graph_def()

# Utilizar tf2onnx para procesar el grafo congelado
from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants

# Extra opset para contrib (si es necesario)
extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]

with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_graph_def, name='')

with tf_loader.tf_session(graph=tf_graph) as sess:
    onnx_graph = process_tf_graph(
        tf_graph, 
        input_names=input_names, 
        output_names=output_names, 
        extra_opset=extra_opset
    )

# Optimizar el grafo ONNX
onnx_graph = optimize_graph(onnx_graph)
model_proto = onnx_graph.make_model("converted")

# Guardar el modelo ONNX resultante
utils.save_protobuf("model2b.onnx", model_proto)
print("Conversion complete!")