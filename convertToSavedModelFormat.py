import tensorflow as tf
model = tf.keras.models.load_model('mobilenetv2.h5')
tf.saved_model.save(model,'mobilenetv2_savedmodel')