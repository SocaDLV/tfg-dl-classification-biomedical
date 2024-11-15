import tensorflow as tf

def fine_tune_model(model, train_data, val_data, fine_tune_at=100, epochs=3, initial_lr=1e-5, callbacks=None):
    # Descongelar el modelo base parcialmente
    base_model = model.layers[0]
    base_model.trainable = True

    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Compilar el modelo con una tasa de aprendizaje baja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='fine_tuned_best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Entrenamiento para fine-tuning
    history_fine = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        callbacks=callbacks
    )

    return history_fine
