from pathlib import Path

NUM_TRANSF_EPOCHS=5 # Number of training epochs for transferleaning   ORIGINAL era 20  ->5
NUM_FINETUNE_EPOCHS=2 # Number of training epochs for finetuning steps ORIGINAL era 5  ->2

HIDDEN=256    # Number of nodes in hidden layer
DROPOUT=0.15  # optional dropout rate
MODEL_FILE= Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v2\v1_webMedium.com\ft_mobilenetv2_tinyin.h5') # where to save the trained model
# if we have a saved model, should it get retrained
RETRAIN=False
NUM_RETRAIN_EPOCHS=2              # ORIGINAL era 3 ->2

DIMX=224 # resize images in x-dimension
DIMY=224 # resize images in y-dimension

# Range of image pixel values
IMG_SCALE_MIN = -1.0              #De -1.0 a 1.0!!!       ORIGINALMENT era de -1.0 a 1.0, si gastem els wheights de imagenet
IMG_SCALE_MAX = 1.0

BATCH_SIZE = 32                   # batch size of training data   ORIGINALMENT era de 16, pero es massa poc.

SHUFFLE_BUFFER_SIZE = 1000        # Nova variable pal tamany del buffer