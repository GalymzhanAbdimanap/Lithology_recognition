import os
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import load_model

# Use GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# -----------------------------
# Paths and parameters configuration
# -----------------------------
MODEL_PATH = "models/convnext_best.h5"
TEST_DIR = "data/lithology_filled_dataset/test"
IMG_SIZE = (224, 224)
BATCH_SIZE = 64

# -----------------------------
# Load test dataset
# -----------------------------
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

# -----------------------------
# Define class names (override if needed)
# -----------------------------
class_names = [
    "Blank",
    "Coarse-grained sandstone",
    "Medium-grained sandstone",
    "Fine-grained sandstone",
    "Shaly sandstone",
    "Clay",
    "Coal",
    "Dense rock",
    "Desctructed core"  # class 999
]

# -----------------------------
# Define custom LayerScale layer (required for loading the model)
# -----------------------------
class LayerScale(tf.keras.layers.Layer):
    def __init__(self, projection_dim, init_values=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.init_values = init_values

    def build(self, input_shape):
        self.scale = self.add_weight(
            shape=(self.projection_dim,),
            initializer=tf.keras.initializers.Constant(self.init_values),
            trainable=True,
            name="layer_scale_weight"
        )

    def call(self, inputs):
        return inputs * self.scale

# -----------------------------
# Load trained model with custom layer
# -----------------------------
model = load_model(MODEL_PATH, custom_objects={"LayerScale": LayerScale})

# -----------------------------
# Make predictions on test dataset
# -----------------------------
true_labels = []
pred_labels = []

for images, labels in test_dataset:
    preds = model.predict(images)
    pred_classes = np.argmax(preds, axis=1)
    true_labels.extend(labels.numpy())
    pred_labels.extend(pred_classes)

# -----------------------------
# Confusion matrix calculation and visualization
# -----------------------------
def plot_confusion_matrix(cm, class_names, out_filename, normalize=True):
    idx_999 = class_names.index('Desctructed core')
    new_order = [idx_999] + [i for i in range(len(class_names)) if i != idx_999]

    cm = cm[np.ix_(new_order, new_order)]
    class_names = [class_names[i] for i in new_order]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm * 100, annot=True, fmt='.2f' if normalize else 'd', cmap='magma',
                xticklabels=class_names, yticklabels=class_names, square=True, cbar=True, annot_kws={"size": 8})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.tight_layout()
    plt.savefig(out_filename)
    plt.close()

cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:\n", cm)

plot_confusion_matrix(cm, class_names, 'saved_models_convnext/conf_matrix_convnext_curr.png', normalize=True)

# -----------------------------
# Print classification report
# -----------------------------
print("Classification Report:")
print(classification_report(true_labels, pred_labels, target_names=class_names))
