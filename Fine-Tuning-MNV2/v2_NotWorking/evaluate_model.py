import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, val_data):
    # Evaluar el modelo y mostrar pérdida y precisión
    loss, accuracy = model.evaluate(val_data, verbose=1)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy * 100:.2f}%')

    # Obtener predicciones y etiquetas reales
    val_data.reset()  # Reinicia el generador
    y_pred = model.predict(val_data, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = val_data.classes

    # Generar un reporte de clasificación
    class_labels = list(val_data.class_indices.keys())
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=class_labels))

    # Crear y mostrar la matriz de confusión
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
