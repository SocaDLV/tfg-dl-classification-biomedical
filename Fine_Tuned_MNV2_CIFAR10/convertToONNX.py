import tensorflow as tf
import tf2onnx
import onnx

model = tf.keras.models.load_model(r'C:\Users\...\mnv2_cifar10_bo_v1.h5')

full_model = tf.function(lambda inputs: model(inputs))    
full_model = full_model.get_concrete_function([tf.TensorSpec(model_input.shape, model_input.dtype) for model_input in model.inputs])

input_names = [inp.name for inp in full_model.inputs]
output_names = [out.name for out in full_model.outputs]
print("Inputs:", input_names)
print("Outputs:", output_names)

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
frozen_func = convert_variables_to_constants_v2(full_model)
frozen_func.graph.as_graph_def()

from tf2onnx import tf_loader
from tf2onnx.tfonnx import process_tf_graph
from tf2onnx.optimizer import optimize_graph
from tf2onnx import utils, constants
from tf2onnx.handler import tf_op
extra_opset = [utils.make_opsetid(constants.CONTRIB_OPS_DOMAIN, 1)]

with tf.Graph().as_default() as tf_graph:
    tf.import_graph_def(frozen_func.graph.as_graph_def(), name='')
with tf_loader.tf_session(graph=tf_graph):
    g = process_tf_graph(tf_graph, input_names=input_names, output_names=output_names, extra_opset=extra_opset)

onnx_graph = optimize_graph(g)
model_proto = onnx_graph.make_model("converted")
utils.save_protobuf("model2b.onnx", model_proto)

print("Conversion complete!")