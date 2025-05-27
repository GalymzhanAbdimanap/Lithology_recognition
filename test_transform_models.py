"""
Lithology Classification using ViT/Swin Transformer

This script loads a pre-trained ViT/Swin Transformer model and evaluates its performance on lithological image classification. 
It also provides functionality to visualize predictions and confusion matrix.
"""

import os
import numpy as np
import cv2
from glob import glob
import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

# Set visible GPU (change to "0" or "1" depending on your environment)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Hyperparameters and configuration
hp = {
    "image_size": 224,
    "num_channels": 3,
    "input_shape": (224, 224, 3),
    "patch_size": 25,
    "batch_size": 32,
    "lr": 1e-4,
    "num_epochs": 100,
    "num_classes": 9,
    "class_names": [
        "Blank",
        "Coarse-grained sandstone",
        "Medium-grained sandstone",
        "Fine-grained sandstone",
        "Shaly sandstone",
        "Clay",
        "Coal",
        "Dense rock",
        "Desctructed core"
    ]
}

# ViT-specific parameters (if needed)
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])
hp["num_layers"] = 12
hp["hidden_dim"] = 384
hp["mlp_dim"] = 1536
hp["num_heads"] = 6
hp["dropout_rate"] = 0.1


def create_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def load_data(path):
    """Load test image paths."""
    test_x = glob(os.path.join(path, "test", "*", "*.jpg"))
    return test_x


def process_image_label(path, image_size=(224, 224)):
    """Read image, resize, normalize, and extract class label."""
    image = cv2.imread(path.decode(), cv2.IMREAD_COLOR)
    image = cv2.resize(image, image_size)
    image = image / 255.0
    image = image.astype(np.float32)

    class_name = path.decode().split("/")[-2]
    class_idx = np.array(int(class_name), dtype=np.int32)
    if class_name == '999':
        class_idx = np.array(int('8'), dtype=np.int32)

    return image, class_idx


def parse(path):
    """TensorFlow parsing function for dataset pipeline."""
    image, label = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])
    label = tf.one_hot(label, depth=hp["num_classes"])
    image.set_shape((hp["image_size"], hp["image_size"], hp["num_channels"]))
    label.set_shape((hp["num_classes"],))
    return image, label


def tf_dataset(images, batch=32):
    """Create TensorFlow dataset from image paths."""
    ds = tf.data.Dataset.from_tensor_slices(images)
    ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def evaluate_model(model, test_ds, class_names, out_filename):
    """Evaluate model on test set and print classification report."""
    y_true, y_pred, y_pred_np = [], [], []

    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))
        y_pred_np.extend(preds)

    np.save('transformer_predict.npy', y_pred_np)
    np.save('transformer_true.npy', y_true)

    cm = confusion_matrix(y_true, y_pred)
    print(cm)

    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:\n", report)

    plot_confusion_matrix(cm, class_names, out_filename)


def plot_confusion_matrix(cm, class_names, out_filename, normalize=False):
    """Plot and save confusion matrix (optionally normalized)."""
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


def predict_image(model, image_path, class_names, image_size=(224, 224)):
    """Predict class of a single image."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, image_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    preds = model.predict(image)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    print(f"Predicted Class: {class_names[pred_class]}, Confidence: {confidence:.4f}")
    return class_names[pred_class], confidence


if __name__ == "__main__":
    dataset_path = "data/lithology_filled_dataset_copy"
    test_x = load_data(dataset_path)
    test_ds = tf_dataset(test_x, batch=hp["batch_size"])

    # Choose model path(s)
    model_paths = [
        'models/vit_model.100-2.7675.tf',
        # 'files_pretrain_swin_balanced_dataset/model.72-5.3924.tf',
        # Add other models if needed
    ]

    for best_model_path in model_paths:
        out_filename = f'{best_model_path.split("/")[0]}.png'
        model = tf.keras.models.load_model(best_model_path, custom_objects={'KerasLayer': hub.KerasLayer})

        evaluate_model(model, test_ds, hp["class_names"], out_filename)

    # Example prediction
    # sample_image_path = test_x[0]
    # predict_image(model, sample_image_path, hp["class_names"])
