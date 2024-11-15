import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prepare_data(train_dir, val_dir, img_size=(64, 64), batch_size=32):
    # Data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    val_generator = datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='sparse'
    )

    return train_generator, val_generator