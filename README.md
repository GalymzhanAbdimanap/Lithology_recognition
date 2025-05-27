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
Prepare your dataset([download](https://drive.google.com/drive/folders/1jKMcqHT8ODdCtwz6fX75S51zGS7eoHPj?usp=sharing)) with the following directory structure for testing:
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
## Dataset preview 
| Image | Class |
|-----------|-----------|
|![0](images/0.jpg)           |  Blank          |
|![1](images/1.jpg)           |  Coarse-grained sandstone         |
|![2](images/2.jpg)           |  Medium-grained sandstone          |
|![3](images/3.jpg)           |  Fine-grained sandstone          |
|![4](images/4.jpg)           |  Shaly sandstone         |
|![5](images/5.jpg)           |  Clay         |
|![6](images/6.jpg)           |  Coal          |
|![7](images/7.jpg)           |  Dense rock         |
|![999](images/999.jpg)           |  Desctructed core         |


## How to Run
Each script loads the corresponding trained model from the models folder and evaluates it on the test dataset:

1. Evaluate ConvNeXt model
```bash
python test_convnext.py
```

Loads ConvNeXt model with custom LayerScale layer.  
Generates predictions and prints classification report.  
Saves normalized confusion matrix plot to models/.  

2. Evaluate EfficientNetV2 model
```bash
python test_efficientnetv2.py
```
Loads EfficientNetV2-based model.  
Performs evaluation on test dataset.  
Saves confusion matrix plot and classification report.  

3. Evaluate Transformer-based models
```bash
python test_transform_models.py
```
Loads transformer-based models.  
Performs evaluation on test dataset  
Computes and saves combined confusion matrix and classification report.  

## Notes
Ensure your GPU is available or modify scripts to run on CPU.  
Adjust paths in the scripts if your dataset or models are stored elsewhere.  
The class names are consistent across all scripts:  
```bash
["Blank", "Coarse-grained sandstone", "Medium-grained sandstone", 
 "Fine-grained sandstone", "Shaly sandstone", "Clay", "Coal", 
 "Dense rock", "Desctructed core"]
```
