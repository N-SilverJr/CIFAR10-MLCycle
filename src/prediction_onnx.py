"""
ONNX-based prediction for lightweight cloud deployment.
Uses onnxruntime instead of TensorFlow for inference.

Preprocessing is duplicated here (not imported from src.preprocessing)
because that module imports tensorflow.keras.utils at the top level.
requirements-render.txt deliberately excludes TensorFlow to stay within
memory limits, so pulling in src.preprocessing would crash this deployment.
Keep this file's preprocessing in sync with src/preprocessing.py.
"""
import io

import numpy as np
import onnxruntime as ort
from PIL import Image

IMAGE_SIZE = (32, 32)
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CONFIDENCE_THRESHOLD = 0.45


def _preprocess_image_array(img):
    """Convert PIL image to normalized (1, 32, 32, 3) float32 array."""
    img = img.convert("RGB").resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32") / 255.0
    return arr.reshape(1, 32, 32, 3)


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    return _preprocess_image_array(img)


def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return session


def predict_single_onnx(session, image_bytes):
    processed = preprocess_image(image_bytes).astype(np.float32)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: processed})[0]
    probabilities = output[0]

    # Softmax if the export produced logits
    if np.any(probabilities < 0) or np.abs(np.sum(probabilities) - 1.0) > 0.01:
        exp_preds = np.exp(probabilities - np.max(probabilities))
        probabilities = exp_preds / np.sum(exp_preds)

    predicted_idx = int(np.argmax(probabilities))
    conf = float(probabilities[predicted_idx])
    in_domain = conf >= CONFIDENCE_THRESHOLD
    result = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": conf,
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
        "in_domain": in_domain,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "warning": None if in_domain else (
            "Low confidence — this image may not belong to any CIFAR-10 class "
            "(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). "
            f"Best guess is shown below ({CLASS_NAMES[predicted_idx]}, {conf*100:.1f}%)."
        ),
    }
    return result
