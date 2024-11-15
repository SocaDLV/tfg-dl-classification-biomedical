import tensorflow as tf

def train_model(model, train_data, val_data, epochs=5, initial_lr=0.001, callbacks=None):
    # Define el optimizador con tasa de aprendizaje inicial
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    
    # Compilación del modelo con el optimizador y la pérdida
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks
    callbacks = [
        # Reduce la tasa de aprendizaje si no hay mejoras en la precisión
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1,
            min_lr=1e-6
        ),
        # Parada temprana si no hay mejoras en la validación
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        # Guardar el modelo con mejor rendimiento
        tf.keras.callbacks.ModelCheckpoint(
            filepath='best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # Entrenamiento del modelo
    history = model.fit(
        train_data,
        epochs=epochs,
        validation_data=val_data,
        steps_per_epoch=1000,
        callbacks=callbacks,
        verbose=1
    )
    
    return history
