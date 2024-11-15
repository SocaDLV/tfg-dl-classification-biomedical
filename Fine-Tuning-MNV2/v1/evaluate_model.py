def evaluate_model(model, val_data):
    loss, accuracy = model.evaluate(val_data)
    print(f'Validation accuracy: {accuracy * 100:.2f}%')