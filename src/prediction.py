"""Prediction helpers shared by the API and batch/demo scripts."""
from src.preprocessing import preprocess_image, preprocess_image_from_path, CLASS_NAMES


def _predict_from_array(model, image_array):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_idx = int(probabilities.argmax())
    return {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
    }


def predict_single(model, image_bytes):
    """Predict a CIFAR-10 class from raw uploaded image bytes."""
    image_array = preprocess_image(image_bytes)
    return _predict_from_array(model, image_array)


def predict_from_path(model, image_path):
    """Predict a CIFAR-10 class from an image file on disk."""
    image_array = preprocess_image_from_path(image_path)
    return _predict_from_array(model, image_array)
