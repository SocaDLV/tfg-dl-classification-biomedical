import tensorflow as tf

def prepare_data(train_dir, val_dir, img_size=(64, 64), batch_size=32):
    # Define preprocesamiento base
    preprocess = tf.keras.layers.Rescaling(1./255)

    # Data augmentation solo para entrenamiento
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomWidth(0.2),
        tf.keras.layers.RandomHeight(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomFlip("horizontal")
    ])

    # Crear dataset de entrenamiento con augmentación
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size
    )
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(preprocess(x)), y))
    
    # Crear dataset de validación (sin augmentación)
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size
    )
    val_dataset = val_dataset.map(lambda x, y: (preprocess(x), y))

    # Prefetch para mejorar el rendimiento
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset
