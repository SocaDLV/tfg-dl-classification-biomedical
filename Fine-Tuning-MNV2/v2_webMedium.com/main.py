from plot import show_images, plot_training_history, concatenate_histories
import sys, os, datetime
from pathlib import Path
import tensorflow as tf

from const import NUM_TRANSF_EPOCHS, NUM_FINETUNE_EPOCHS, MODEL_FILE, NUM_RETRAIN_EPOCHS, RETRAIN
from load import load_tinyimagenet, build_datasets
from ft_net import build_ft_net, compile_model, unfreeze_base_layers
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def main():
    print("Tensorflow version: ", tf.__version__)
    print("Python version: ", sys.version)

    # Set the log directory to store the logs for tensorboard
    log_dir = os.path.join(
        "logs", "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=1)

    # Cargar y preprocesar el dataset Tiny ImageNet
    data_dir = ""  # Ruta al directorio del dataset
    train_images, train_labels, val_images, val_labels, class_names = load_tinyimagenet(data_dir)

    # Construir datasets de entrenamiento, validación y prueba
    train_dataset, validation_dataset = build_datasets(train_images, train_labels, val_images, val_labels)

    fit_history_t = None
    fit_history_retr = None
    learning_rate = 0.001

    if not Path(MODEL_FILE).is_file():  # No stored model -> Build the FT model
        model = build_ft_net(out_dim=len(class_names), learning_rate=learning_rate)
        fit_history_t = model.fit(train_dataset, epochs=NUM_TRANSF_EPOCHS, validation_data=validation_dataset, callbacks=[tensorboard_callback])
        loss, accuracy = model.evaluate(validation_dataset)
        print("TRANSFER LEARN: validation accuracy: ", accuracy, "validation loss: ", loss)
        model.save(MODEL_FILE)

    else:  # we have a stored model
        print("You already have a model, erase it or save it")
        """"
        model = tf.keras.models.load_model(MODEL_FILE)
        if RETRAIN:
            compile_model(model, learning_rate=learning_rate / 2.)
            # Further train the model
            fit_history_retr = model.fit(train_dataset, epochs=NUM_RETRAIN_EPOCHS,
                                         validation_data=validation_dataset, callbacks=[tensorboard_callback])
            model.save(MODEL_FILE)
        """

    # Start Fine Tuning
    learning_rate = 0.0001
    unfreeze = 92
    ##model = unfreeze_base_layers(model, layers=unfreeze, learning_rate=learning_rate) -> !!Primer arreglar metodo en ft_net!!
    fit_history_ft1 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
                                validation_data=validation_dataset, callbacks=[tensorboard_callback])
    loss, accuracy = model.evaluate(validation_dataset)
    print("FINETUNE Step 1: validation accuracy: ", accuracy, "validation loss: ", loss)

    unfreeze += 9
    model = unfreeze_base_layers(model, layers=unfreeze, learning_rate=learning_rate / 2.0)
    fit_history_ft2 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
                                validation_data=validation_dataset, callbacks=[tensorboard_callback])
    loss, accuracy = model.evaluate(validation_dataset)
    print("FINETUNE Step 2: validation accuracy: ", accuracy, "validation loss: ", loss)

    unfreeze += 9
    model = unfreeze_base_layers(model, layers=unfreeze, learning_rate=learning_rate / 4.0)
    fit_history_ft3 = model.fit(train_dataset, epochs=NUM_FINETUNE_EPOCHS,
                                validation_data=validation_dataset, callbacks=[tensorboard_callback])
    loss, accuracy = model.evaluate(validation_dataset)
    print("FINETUNE Step 3: validation accuracy: ", accuracy, "validation loss: ", loss)

    concatenated_history = concatenate_histories(
        fit_history_t, fit_history_retr, fit_history_ft1, fit_history_ft2, fit_history_ft3)
    plot_training_history("Finetuned ft_net", concatenated_history)

    # Plot some predictions
    pred_dataset = validation_dataset.take(1).cache()
    predictions = model.predict(pred_dataset)
    max_predictions = tf.argmax(predictions, axis=1)
    show_images(pred_dataset, "Tiny ImageNet Predictions", 5, class_names,
                one_hot=False, predicted_classes=max_predictions, figsize=25, predictions=predictions)
    model.save(Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\v2_webMedium.com\ft_mobilenetv2_tinyin.h5'))

if __name__ == "__main__":
    main()