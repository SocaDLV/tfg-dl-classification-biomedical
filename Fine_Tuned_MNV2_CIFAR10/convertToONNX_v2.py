import tensorflow as tf
import onnx
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model_path = r'C:\Users\...\mnv2_cifar10_bo_v1.h5'
model = tf.keras.models.load_model(model_path)

input_shape = (1, 224, 224, 3)  
input_signature = tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="input")

full_model = tf.function(lambda inputs: model(inputs))
concrete_function = full_model.get_concrete_function(input_signature)

frozen_func = convert_variables_to_constants_v2(concrete_function)
frozen_graph_def = frozen_func.graph.as_graph_def()

with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_graph_def, name='')

input_names = [frozen_func.inputs[0].name]  
output_names = [frozen_func.outputs[0].name]  

print("Input names:", input_names)
print("Output names:", output_names)

from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants

extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]

with tf_loader.tf_session(graph=tf_graph) as sess:
    onnx_graph = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)

onnx_graph = optimize_graph(onnx_graph)
model_proto = onnx_graph.make_model("converted")

onnx.save_model(model_proto, "model2b.onnx")
print("Conversión a ONNX completada!")
