import tensorflow as tf

def fine_tune_model(model, train_data, val_data, fine_tune_at=100, epochs=1):  #orig: 10 epochs
    # Unfreeze some layers for fine-tuning
    base_model = model.layers[0]
    base_model.trainable = True
    
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Recompile the model for fine-tuning
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Continue training
    history_fine = model.fit(train_data,
                             epochs=epochs,
                             steps_per_epoch=1000,
                             validation_data=val_data)
    
    return history_fine