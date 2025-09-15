Brief, reproducible assets for a final-degree project benchmarking deep-learning image classification across heterogeneous edge hardware, plus a small biomedical demo.

Repository contents

- Introductory_Test/
- - Dataset, per-hardware inference code, conversion scripts, and the MobileNetV2 model in all formats needed for inference.

- Fine_Tuned_(NameOfTheModel)_CIFAR10/
- - Inference code, trained models, and conversion scripts to reproduce all experiments in “Benchmarking fine-tuned models.”

- Training_Notebooks/
- - The .ipynb notebooks used for model training and fine-tuning.

- Ranking_Performance_Fine_Tuned_CIFAR10.py
- - Python script that computes the ranking used in “Benchmarking fine-tuned models,” with the final data already filled in.

- Biomedical_Application/
- - Inference code and correctly converted models to reproduce the biomedical test.

- Important notes

- - Pick the highest versioned file inside each code folder (e.g., V3 > V2, 3 > 1). The highest number is the most optimized version of that test.

- - Code comments were intentionally removed and paths simplified to make reading easier and adaptation to other environments straightforward.

- - Paths to datasets/models are minimal and may need to be adjusted to your local setup.