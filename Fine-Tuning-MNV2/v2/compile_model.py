import tensorflow as tf

def compile_model(model, initial_lr=0.001):
    # Definir un programador de tasa de aprendizaje (opcional)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_lr,
        decay_steps=10000,
        decay_rate=0.9,
        staircase=True
    )

    # Usar un optimizador avanzado
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Compilar el modelo con métrica de precisión y regularización
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
