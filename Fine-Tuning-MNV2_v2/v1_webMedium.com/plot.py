from const import IMG_SCALE_MIN, IMG_SCALE_MAX
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
matplotlib.use('TkAgg')

def plot_image(ax, image, label, cmap='gray_r', title=None, title_color='black'):
    """Helper function to plot a single image with label and optional title."""
    ax.imshow(image, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    if title:
        ax.set_xlabel(title, color=title_color)
    else:
        ax.set_xlabel(label)


def show_images(dataset: tf.data.Dataset, title: str,
                tile_dim: int, class_names: list[str],
                one_hot: bool, predicted_classes: list[int] | None = None,
                figsize: int | float = 10, predictions: np.array = None):
    """
    Displays a grid of images from a TensorFlow dataset.

    Parameters:
    dataset (tf.data.Dataset): The dataset containing the images and labels.
    title (str): The title for the plot.
    tile_dim (int): The dimension of the grid (e.g., tile_dim=4 will create a 4x4 grid).
    class_names (list[str]): List of class names corresponding to the labels.
    one_hot (bool): Whether the labels are one-hot encoded.
    predicted_classes (list[int] | None): Optional list of predicted class indices for the images. Defaults to None.
    figsize (int | float): The size of the figure. Defaults to 10.
    predictions (np.array | None): Optional array of prediction probabilities/scores. Defaults to None.

    Returns:
    None
    """
    """
    # unbatch and rebatch the dataset
    try:
        to_plot = dataset.rebatch(tile_dim*tile_dim)
    except:
        to_plot = dataset.batch(tile_dim*tile_dim)
    """ #CANVI PROPOST PER GPT AVALL

    # Check for rebatch support and fallback to batch if unavailable
    if hasattr(dataset, 'rebatch'):
        to_plot = dataset.rebatch(tile_dim * tile_dim)
    else:
        to_plot = dataset.batch(tile_dim * tile_dim)


    # Backscale of image pixels
    img_scale_delta = IMG_SCALE_MAX-IMG_SCALE_MIN

    plt.figure(figsize=(figsize, figsize))
    for images, labels in to_plot.take(1):
        for i in range(tile_dim * tile_dim):
            ax = plt.subplot(tile_dim, tile_dim, i + 1)
            img = (images[i].numpy().squeeze()-IMG_SCALE_MIN)/img_scale_delta
            # Determine label index
            label_index = np.argmax(labels[i]) if one_hot else labels[i]

            if predicted_classes is None:
                plot_image(
                    ax, img, class_names[label_index], cmap=plt.cm.gray_r)
            else:
                pred_label = class_names[predicted_classes[i]]
                if predictions is not None:
                    pred_value = predictions[i, predicted_classes[i]]
                    pred_label += f"/{pred_value*100.0:.1f}%"
                true_label = class_names[label_index]
                
                #title_color = "red" if predicted_classes[i] != label_index else "black" UNA SOLUCIÓ BAIX
                #title_color = "red" if tf.constant(predicted_classes[i], dtype=tf.int64) != label_index else "black"
                title_color = "red" if tf.constant(predicted_classes[i], dtype=tf.int64).numpy() != label_index.numpy() else "black"


                plot_image(
                    ax, img, class_names[label_index], title=f"Pred: {pred_label}, True: {true_label}", title_color=title_color)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.show()


def concatenate_histories(*histories: tuple[tf.keras.callbacks.History]) -> tf.keras.callbacks.History:
    """
    Concatenates the training histories from multiple Keras model fits into a single history object.

    Parameters:
    *histories (tf.keras.callbacks.History): Variable number of Keras History objects from model fits.
                                             Histories that are None are ignored.

    Returns:
    tf.keras.callbacks.History: A new History object containing the concatenated training history.

    Raises:
    ValueError: If no valid histories are provided.
    """
    # Filter out None histories and extract the history dictionaries
    valid_histories = [
        history.history for history in histories if history is not None]

    if not valid_histories:
        raise ValueError("No valid histories provided")

    # Initialize a dictionary to hold the concatenated history
    concatenated_history = {}

    # Iterate over the keys in the first valid history to concatenate all histories
    for key in valid_histories[0].keys():
        concatenated_history[key] = []
        for history in valid_histories:
            concatenated_history[key].extend(history[key])

    # Create a new History object to return
    ret_history = tf.keras.callbacks.History()
    ret_history.history = concatenated_history

    return ret_history


def plot_training_history(title: str, concatenated_history: tf.keras.callbacks.History):
    """
    Plots the training and validation loss and accuracy from a concatenated Keras training history.

    Parameters:
    title (str): The title for the plots.
    concatenated_history (tf.keras.callbacks.History or dict): The concatenated training history. 
                                                              This can be a History object or a dictionary containing 
                                                              the training and validation metrics.

    Returns:
    None
    """
    # Extract history if it's wrapped in an object with a .history attribute
    if hasattr(concatenated_history, 'history'):
        concatenated_history = concatenated_history.history

    # Determine the number of epochs
    epochs = range(1, len(concatenated_history['loss']) + 1)

    # Plot the training and validation loss
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, concatenated_history['loss'], label='Training Loss')
    plt.plot(epochs, concatenated_history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(np.arange(1, len(epochs) + 1, step=1))
    plt.ylim(0.0, 0.7)  # Set y-axis limits for loss

    # Plot the training and validation accuracy (if available)
    if 'accuracy' in concatenated_history and 'val_accuracy' in concatenated_history:
        plt.subplot(1, 2, 2)
        plt.plot(
            epochs, concatenated_history['accuracy'], label='Training Accuracy')
        plt.plot(
            epochs, concatenated_history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.xticks(np.arange(1, len(epochs) + 1, step=1))
        plt.ylim(0.7, 1.0)  # Set y-axis limits for accuracy

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()