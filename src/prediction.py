"""Prediction helpers shared by the API and batch/demo scripts."""
from src.preprocessing import preprocess_image, preprocess_image_from_path, CLASS_NAMES

# Below this top-class probability we treat the input as likely out-of-domain.
CONFIDENCE_THRESHOLD = 0.45


def _attach_domain_flags(result):
    """Flag low-confidence predictions as possibly outside CIFAR-10."""
    conf = result["confidence"]
    in_domain = conf >= CONFIDENCE_THRESHOLD
    result["in_domain"] = in_domain
    result["confidence_threshold"] = CONFIDENCE_THRESHOLD
    if in_domain:
        result["warning"] = None
    else:
        result["warning"] = (
            "Low confidence — this image may not belong to any CIFAR-10 class "
            "(airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck). "
            f"Best guess is shown below ({result['predicted_class']}, {conf*100:.1f}%)."
        )
    return result


def _predict_from_array(model, image_array):
    probabilities = model.predict(image_array, verbose=0)[0]
    predicted_idx = int(probabilities.argmax())
    result = {
        "predicted_class": CLASS_NAMES[predicted_idx],
        "predicted_index": predicted_idx,
        "confidence": float(probabilities[predicted_idx]),
        "probabilities": {name: float(p) for name, p in zip(CLASS_NAMES, probabilities)},
    }
    return _attach_domain_flags(result)


def predict_single(model, image_bytes):
    """Predict a CIFAR-10 class from raw uploaded image bytes."""
    image_array = preprocess_image(image_bytes)
    return _predict_from_array(model, image_array)


def predict_from_path(model, image_path):
    """Predict a CIFAR-10 class from an image file on disk."""
    image_array = preprocess_image_from_path(image_path)
    return _predict_from_array(model, image_array)
