import torch
from fastai.vision.all import *

def main():

    # Ruta donde guardaste el modelo
    model_path = r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-ResNet_cifar10\modelsPytorch\stage-2-pytorch'

    # Cargar el modelo usando torch.load()
    model = torch.load(model_path, map_location=torch.device('cpu'))

    # Asegurarse de que el DataLoader esté bien configurado
    path = Path(r'C:\Users\Ivan\Desktop\Asignatures5tcarrera\TFG\codi\Fine-Tuning-MNV2_v3_cifar10\aa_PC\content\images')
    dls = ImageDataLoaders.from_folder(path, valid='test')

    # Re-crea el learner y asigna el modelo cargado
    learner = vision_learner(dls, resnet18, metrics=accuracy)
    learner.model = model  # Asignamos el modelo cargado a la red

    # Verificación opcional de las primeras predicciones
    preds, _ = learner.get_preds()
    print(preds)
    
if __name__ == '__main__':
    main()