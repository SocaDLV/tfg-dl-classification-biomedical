import tensorflow as tf

def load_model(num_classes=200, img_shape=(224, 224, 3)):
    # Cargar MobileNetV2 preentrenado con ImageNet
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=img_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar las capas base inicialmente
    base_model.trainable = False
    
    # Construir el modelo usando la API funcional
    inputs = tf.keras.Input(shape=img_shape)
    x = base_model(inputs, training=False)  # Evitar cambios en BatchNorm durante inferencia
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
