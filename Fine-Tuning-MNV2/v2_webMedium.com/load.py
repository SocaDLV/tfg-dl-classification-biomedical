import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from const import DIMX, DIMY, BATCH_SIZE, IMG_SCALE_MIN, IMG_SCALE_MAX
from typing import List, Tuple
from pathlib import Path

def load_tinyimagenet(data_dir: str) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, List[str]]:
    """
    Carga y preprocesa el dataset Tiny ImageNet.

    Args:
        data_dir (str): Ruta al directorio que contiene `tinyImgNet-train` y `tinyImgNet-val`.

    Returns:
        Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, List[str]]:
            - Imágenes de entrenamiento (tf.Tensor).
            - Etiquetas de entrenamiento (tf.Tensor).
            - Imágenes de validación (tf.Tensor).
            - Etiquetas de validación (tf.Tensor).
            - Lista de nombres de clases.
    """
    train_dir = os.path.join(data_dir, Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-train'))
    val_dir = os.path.join(data_dir, Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-val'))

    # Cargar clases
    class_names = sorted(os.listdir(train_dir))
    class_to_index = {name: idx for idx, name in enumerate(class_names)}

    # Cargar datos de entrenamiento
    train_images, train_labels = _load_train_data(train_dir, class_to_index)

    # Cargar datos de validación
    val_images, val_labels = _load_val_data(val_dir, class_to_index)

    return train_images, train_labels, val_images, val_labels, class_names


def _load_train_data(train_dir: str, class_to_index: dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Carga imágenes y etiquetas del directorio de entrenamiento.

    Args:
        train_dir (str): Ruta al directorio de entrenamiento.
        class_to_index (dict): Mapeo de nombres de clases a índices.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Imágenes y etiquetas de entrenamiento.
    """
    images, labels = [], []
    for class_name, class_idx in class_to_index.items():
        class_path = os.path.join(train_dir, class_name, "images")
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [DIMX, DIMY])
            img = img / 255.0  # Normalizar
            images.append(img)
            labels.append(class_idx)

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)


def _load_val_data(val_dir: str, class_to_index: dict) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Carga imágenes y etiquetas del directorio de validación.

    Args:
        val_dir (str): Ruta al directorio de validación.
        class_to_index (dict): Mapeo de nombres de clases a índices.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: Imágenes y etiquetas de validación.
    """
    val_annotations_path = os.path.join(val_dir, "val_annotations.txt")
    val_images_dir = os.path.join(val_dir, "images")

    # Leer anotaciones de validación
    with open(val_annotations_path, "r") as f:
        lines = f.readlines()

    val_map = {}
    for line in lines:
        parts = line.split("\t")
        img_name, class_name = parts[0], parts[1]
        val_map[img_name] = class_name

    images, labels = [], []
    for img_name, class_name in val_map.items():
        img_path = os.path.join(val_images_dir, img_name)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [DIMX, DIMY])
        img = img / 255.0  # Normalizar
        images.append(img)
        labels.append(class_to_index[class_name])

    return tf.convert_to_tensor(images), tf.convert_to_tensor(labels)


def build_datasets(train_images: tf.Tensor, train_labels: tf.Tensor,
                   val_images: tf.Tensor, val_labels: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Construye datasets de entrenamiento y validación para TensorFlow.

    Args:
        train_images (tf.Tensor): Imágenes de entrenamiento.
        train_labels (tf.Tensor): Etiquetas de entrenamiento.
        val_images (tf.Tensor): Imágenes de validación.
        val_labels (tf.Tensor): Etiquetas de validación.

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Datasets de entrenamiento y validación.
    """
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))

    # Barajar y agrupar
    train_dataset = train_dataset.shuffle(len(train_images)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
