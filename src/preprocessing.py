"""Image and dataset preprocessing for CIFAR-10 (32x32 RGB).

Shared by the API, the retraining pipeline, and the notebook.
"""
import io
import os

import numpy as np
from PIL import Image
from tensorflow.keras.utils import to_categorical

IMAGE_SIZE = (32, 32)
NUM_CLASSES = 10
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp")

# CIFAR-10 class names (index == label)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _image_to_array(img):
    """Convert a PIL image to a normalized (32, 32, 3) float32 array in [0, 1]."""
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return arr


def preprocess_image(image_bytes):
    """Preprocess raw uploaded image bytes for model.predict().

    Returns array of shape (1, 32, 32, 3).
    """
    img = Image.open(io.BytesIO(image_bytes))
    arr = _image_to_array(img)
    return arr.reshape(1, 32, 32, 3)


def preprocess_image_from_path(image_path):
    """Preprocess an image file on disk for model.predict()."""
    img = Image.open(image_path)
    arr = _image_to_array(img)
    return arr.reshape(1, 32, 32, 3)


def load_and_preprocess_dataset(data_dir):
    """Load a labeled image dataset from data_dir/{0..9}/*.png style folders.

    Returns (X, y):
        X  shape (n, 32, 32, 3) normalized to [0, 1]
        y  one-hot shape (n, 10)
    """
    images = []
    labels = []

    class_dirs = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.isdigit()
    )

    for class_dir in class_dirs:
        label = int(class_dir)
        if not (0 <= label <= 9):
            continue
        class_path = os.path.join(data_dir, class_dir)
        for filename in sorted(os.listdir(class_path)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue
            img_path = os.path.join(class_path, filename)
            img = Image.open(img_path)
            images.append(_image_to_array(img))
            labels.append(label)

    if not images:
        raise ValueError(f"No valid images found under {data_dir}")

    X = np.array(images, dtype="float32")
    y = to_categorical(np.array(labels), num_classes=NUM_CLASSES)
    return X, y


def preprocess_raw_arrays(X_raw, y_raw):
    """Preprocess raw CIFAR-10-style arrays (uint8 images, integer labels).

    X_raw expected shape (n, 32, 32, 3), y_raw shape (n,) or (n, 1).
    Returns (X, y) with X in [0,1] and y one-hot.
    """
    X = X_raw.astype("float32") / 255.0
    y_flat = y_raw.ravel() if y_raw.ndim > 1 else y_raw
    y = to_categorical(y_flat, num_classes=NUM_CLASSES)
    return X, y
