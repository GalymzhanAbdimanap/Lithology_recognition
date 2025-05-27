import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Use only GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------
# Configuration
# -----------------------------
TEST_DIR = "data/lithology_filled_dataset/test"
MODEL_PATH = "saved_models_lr_0_001/model_best.h5"
OUTPUT_CONF_MATRIX = "saved_models_lr_0_001/conf_matrix.png"

BATCH_SIZE = 64
IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "Blank",
    "Coarse-grained sandstone",
    "Medium-grained sandstone",
    "Fine-grained sandstone",
    "Shaly sandstone",
    "Clay",
    "Coal",
    "Dense rock",
    "Desctructed core"  # class "999" is mapped to this
]

# -----------------------------
# Load Test Dataset
# -----------------------------
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# -----------------------------
# Load Trained Model
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -----------------------------
# Predict on Test Set
# -----------------------------
y_true = np.concatenate([y.numpy() for _, y in test_dataset], axis=0)
y_pred_probs = model.predict(test_dataset)
y_pred = np.argmax(y_pred_probs, axis=1)

# Save predictions and ground truth
np.save('cnn_predict.npy', y_pred_probs)
np.save('cnn_true.npy', y_true)

# -----------------------------
# Confusion Matrix Plotting
# -----------------------------
def plot_confusion_matrix(cm, class_names, out_filename, normalize=True):
    """
    Plot and save the confusion matrix with optional normalization.
    Moves 'Desctructed core' (class '999') to the top of the matrix.
    """
    idx_999 = class_names.index('Desctructed core')
    new_order = [idx_999] + [i for i in range(len(class_names)) if i != idx_999]
    
    cm = cm[np.ix_(new_order, new_order)]
    class_names = [class_names[i] for i in new_order]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaNs (in case of empty class) with 0

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm * 100, annot=True, fmt='.2f' if normalize else 'd', cmap='magma',
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar=True, annot_kws={"size": 8})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()

# -----------------------------
# Evaluation
# -----------------------------
conf_matrix = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Save confusion matrix plot
plot_confusion_matrix(conf_matrix, CLASS_NAMES, OUTPUT_CONF_MATRIX, normalize=True)

# Print classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
