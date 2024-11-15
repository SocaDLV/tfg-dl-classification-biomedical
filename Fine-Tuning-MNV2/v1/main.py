from prepare_data import prepare_data
from load_model import load_model
from compile_model import compile_model
from train_model import train_model
from fine_tune_model import fine_tune_model
from evaluate_model import evaluate_model

from pathlib import Path

# Define directories and parameters
train_dir = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-train'))  # Update with your training directory path
val_dir = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\tinyImgNet-val'))      # Update with your validation directory path
model_save_path = (Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\v1\fine_tuned_mobilenetv2.keras'))  # Path to save the model

# Prepare data
train_generator, val_generator = prepare_data(train_dir, val_dir)

# Load and compile the model
model = load_model()
compile_model(model)

# Train the model
train_history = train_model(model, train_generator, val_generator)

# Fine-tune the model
fine_tune_history = fine_tune_model(model, train_generator, val_generator)

# Evaluate the model
evaluate_model(model, val_generator)

# Save the fine-tuned model
model.save(model_save_path)
print(f'Model saved to {model_save_path}')