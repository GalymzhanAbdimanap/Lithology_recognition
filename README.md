# Lithology Classification Models Evaluation

This repository contains evaluation scripts for various deep learning models used to classify lithology images.

## Project Structure
```bash
models/ # Folder containing saved model files
test_convnext.py # Script to evaluate ConvNeXt model
test_efficientnetv2.py # Script to evaluate EfficientNetV2 model
test_transform_models.py # Script to evaluate transformer-based models
```

---

## Requirements

- Python 3.8+
- TensorFlow
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install dependencies via:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

Dataset Structure
Prepare your dataset with the following directory structure for testing:
```bash
data/
└── lithology_filled_dataset/
    └── test/
        ├── (1)Blank/
        ├── (2)Coarse-grained sandstone/
        ├── (3)Medium-grained sandstone/
        ├── (4)Fine-grained sandstone/
        ├── (5)Shaly sandstone/
        ├── (6)Clay/
        ├── (7)Coal/
        ├── (8)Dense rock/
        └── (999)Desctructed core/
```
How to Run
Each script loads the corresponding trained model from the models folder and evaluates it on the test dataset:

1. Evaluate ConvNeXt model
```bash
python test_convnext.py
```

Loads ConvNeXt model with custom LayerScale layer.<p>
Generates predictions and prints classification report.<p>
Saves normalized confusion matrix plot to models/.<p>

2. Evaluate EfficientNetV2 model
```bash
python test_efficientnetv2.py
```
Loads EfficientNetV2-based model.<p>
Performs evaluation on test dataset.<p>
Saves confusion matrix plot and classification report.<p>

3. Evaluate Transformer-based models
```bash
python test_transform_models.py
```
Loads transformer-based models.<p>
Combines predictions from multiple models (e.g., CNN + Transformer).<p>
Computes and saves combined confusion matrix and classification report.<p>

Notes
Ensure your GPU is available or modify scripts to run on CPU.<p>
Adjust paths in the scripts if your dataset or models are stored elsewhere.<p>
The class names are consistent across all scripts:<p>
```bash
["Blank", "Coarse-grained sandstone", "Medium-grained sandstone", 
 "Fine-grained sandstone", "Shaly sandstone", "Clay", "Coal", 
 "Dense rock", "Desctructed core"]
```
