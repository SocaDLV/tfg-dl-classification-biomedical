import tensorflow as tf
from const import DIMX, DIMY, HIDDEN, DROPOUT

def build_ft_net(out_dim: int, learning_rate: float) -> tf.keras.Model:
    """
    Constructs and compiles a MobileNetV2-based model with a modified 
    top layer suitable for a specific classification task. 
    This function initializes a MobileNetV2 model pre-trained on ImageNet 
    without its top layer, adds custom
    layers and compiles the model with specified output dimensions 
    and learning rate.

    Parameters:
    out_dim : int
        The number of output classes for the model. 
        This determines the dimensionality of the output layer.
    learning_rate : float
        The learning rate for the optimizer.

    Returns:
    tf.keras.models.Model
        The compiled Keras model ready for training.

    Note:
    The function assumes the presence of certain global variables such as 
    `DIMX`, `DIMY` for image dimensions, and `HIDDEN` for the number 
    of units in the hidden dense layer. Adjust these variables 
    as needed based on the specific use case and data.
    """

    # Load the base model with imagenet weights without the top layer
    base_model = tf.keras.applications.MobileNetV2(include_top=False, 
        weights='imagenet', input_shape=(DIMX, DIMY, 3))
    base_model.trainable = False  # Freeze the base model initially
    # base_model.summary(show_trainable=True, expand_nested=True)

    # Define input
    input = tf.keras.Input(shape=(DIMX, DIMY, 3), name="input")

    # Add new layers on top of the model
    x = base_model(input, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # optional for your own experiments
    # x = tf.keras.layers.Dropout(DROPOUT)(x) 
    x = tf.keras.layers.Dense(HIDDEN, activation='relu')(x)
    predictions = tf.keras.layers.Dense(out_dim, 
        activation='softmax', name="output")(x)
    model = tf.keras.models.Model(inputs=input, 
        outputs=predictions, name="ft_mobilenetv2_tinyin")
    # model.summary(show_trainable=True, expand_nested=True)

    model = compile_model(model, learning_rate)
    return model


def compile_model(model: tf.keras.Model, learning_rate: float, 
    optimizer=tf.keras.optimizers.Adam) -> tf.keras.Model:
    model.compile(optimizer=optimizer(learning_rate=learning_rate),
      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def unfreeze_base_layers(model: tf.keras.Model, 
    layers: int, learning_rate: float) -> tf.keras.Model:
    """
    Unfreezes a specified number of layers in the base model for fine-tuning.

    Parameters:
    model (tf.keras.Model): The Keras model containing the base model whose layers are to be unfrozen.
    layers (int): The number of top layers in the base model to unfreeze.
    learning_rate (float): The learning rate to use when recompiling the model.

    Returns:
    tf.keras.Model: The updated Keras model with the specified layers unfrozen and the new learning rate applied.
    """
    # Access the base model
    base_model = model.layers[1]

    # Determine the actual number of layers to unfreeze
    num_layers_to_unfreeze = min(len(base_model.layers), layers)
    print(f"Descongelando las últimas {num_layers_to_unfreeze} capas de {len(base_model.layers)} capas totales.")

    # Freeze all layers first (to avoid reactivating unnecessary layers)
    for layer in base_model.layers:
        layer.trainable = False

    # Unfreeze the last `num_layers_to_unfreeze` layers
    for layer in base_model.layers[-num_layers_to_unfreeze:]:
        layer.trainable = True

    # Compile the model again to apply the changes
    model = compile_model(model, learning_rate)
    return model
