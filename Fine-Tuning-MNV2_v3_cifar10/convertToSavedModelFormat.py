import tensorflow as tf
model = tf.keras.models.load_model('Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/mnv2_cifar10_bo_v1.h5')
tf.saved_model.save(model,'Fine-Tuning-MNV2_v3_cifar10/modelosFTuneados/NEW_mnv2_cifar10_bo_v1_saved_model_format')