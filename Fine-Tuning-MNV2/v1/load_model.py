import tensorflow as tf

def load_model(num_classes=200, img_shape=(64, 64, 3)):
    base_model = tf.keras.applications.MobileNetV2(input_shape=img_shape,
                                                   include_top=False,
                                                   weights='imagenet')
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model