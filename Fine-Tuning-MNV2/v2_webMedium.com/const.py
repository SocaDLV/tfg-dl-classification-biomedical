from pathlib import Path

NUM_TRANSF_EPOCHS=3 # Number of training epochs for transferleaning   ORIGINAL era 20
NUM_FINETUNE_EPOCHS=2 # Number of training epochs for finetuning steps ORIGINAL era 5

HIDDEN=256 # Number of nodes in hidden layer
DROPOUT=0.15 # optional dropout rate
MODEL_FILE= Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2\v2_webMedium.com\ft_mobilenetv2_tinyin.h5') # where to save the trained model
# if we have a saved model, should it get retrained
RETRAIN=False
NUM_RETRAIN_EPOCHS=3

DIMX=64 # resize images in x-dimension
DIMY=64 # resize images in y-dimension

# Range of image pixel values
IMG_SCALE_MIN = 0               #De 0 a 1!!!                           ORIGINALMENT era de -1.0 a 1.0
IMG_SCALE_MAX = 1.0

BATCH_SIZE = 32 # batch size of training data