import tensorflow as tf
import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

# Ruta del modelo Keras (.h5)
model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\modelosFTuneados\mnv2_cifar10_bo_v1.h5'
model = tf.keras.models.load_model(model_path)

# Crear un concrete function con batch size = 1
input_shape = (1, 224, 224, 3)  # Asegúrate de que este sea el tamaño correcto de entrada
input_signature = tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")

# Definir la función concreta
full_model = tf.function(lambda inputs: model(inputs))
concrete_function = full_model.get_concrete_function(input_signature)

# Congelar el modelo
frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_graph_def = frozen_func.graph.as_graph_def()

# Crear un grafo de TensorFlow a partir del modelo congelado
with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_graph_def, name='')

# Obtener los nombres de las entradas y salidas del modelo congelado
input_names = [frozen_func.inputs[0].name]  # Nombre del primer tensor de entrada
output_names = [frozen_func.outputs[0].name]  # Nombre del primer tensor de salida

# Mostrar los nombres de entrada y salida
print("Input names:", input_names)
print("Output names:", output_names)

# Procesar el grafo para convertirlo a ONNX
from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants

# Extra opset para contrib (si es necesario)
extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]

with tf_loader.tf_session(graph=tf_graph) as sess:
    onnx_graph = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)

# Optimizar el grafo ONNX
onnx_graph = optimize_graph(onnx_graph)
model_proto = onnx_graph.make_model("converted")

# Guardar el modelo ONNX resultante
onnx.save_model(model_proto, "model2b.onnx")
print("Conversión a ONNX completada!")
