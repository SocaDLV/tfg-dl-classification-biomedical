from prepare_data import prepare_data
from load_model import load_model
from compile_model import compile_model
from train_model import train_model
from fine_tune_model import fine_tune_model
from evaluate_model import evaluate_model

from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.mixed_precision import set_global_policy

# Habilitar Mixed Precision
set_global_policy('mixed_float16')

# Verificar GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Using GPU: {physical_devices[0].name}")
else:
    print("No GPU found, training will use CPU.")

# Define directories and parameters
train_dir = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-train'))  # Update with your training directory path
val_dir = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-val'))      # Update with your validation directory path
model_save_path = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\v2\fine_tuned_mobilenetv2.keras'))  # Path to save the model

# Prepare data
train_generator, val_generator = prepare_data(train_dir, val_dir)

# Load and compile the model
model = load_model()
compile_model(model)

# Define callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ModelCheckpoint(filepath=model_save_path, save_best_only=True, monitor='val_loss', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
    TensorBoard(log_dir="./logs", histogram_freq=1)
]

# Train the model
train_history = train_model(model, train_generator, val_generator, epochs=5, callbacks=callbacks)

# Fine-tune the model
fine_tune_history = fine_tune_model(model, train_generator, val_generator, epochs=3, callbacks=callbacks)

# Evaluate the model
evaluate_model(model, val_generator)

# Save the fine-tuned model
model.save(model_save_path)
print(f'Model saved to {model_save_path}')
