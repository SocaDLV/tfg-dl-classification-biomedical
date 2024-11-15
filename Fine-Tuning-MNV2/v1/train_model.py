def train_model(model, train_data, val_data, epochs=1):  #orig: 20 epochs
    history = model.fit(train_data,
                        epochs=epochs,
                        steps_per_epoch=1000,
                        validation_data=val_data)
    return history