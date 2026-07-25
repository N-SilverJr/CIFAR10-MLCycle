"""
Generate models/eval_metrics.json by evaluating the saved model on the test set.
Run once from the project root after training:
    python scripts/generate_eval_metrics.py
"""
import os
import sys
import json
from datetime import datetime
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import preprocess_raw_arrays, CLASS_NAMES


def main():
    model_path = "models/cifar10_classifier.h5"
    model = keras.models.load_model(model_path)

    X_test_raw = np.load("data/test/X_test.npy")
    y_test_raw = np.load("data/test/y_test.npy")
    X_test, y_test = preprocess_raw_arrays(X_test_raw, y_test_raw)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = y_test_raw.ravel()

    report = classification_report(y_true, y_pred, output_dict=True, target_names=CLASS_NAMES)

    y_train_raw = np.load("data/train/y_train.npy")
    class_distribution = [int(np.sum(y_train_raw.ravel() == i)) for i in range(10)]

    per_class_f1 = [report[name]["f1-score"] for name in CLASS_NAMES]
    cm = confusion_matrix(y_true, y_pred).tolist()

    metrics = {
        "class_names": CLASS_NAMES,
        "class_distribution": class_distribution,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "updated_at": datetime.now().isoformat(),
    }

    os.makedirs("models", exist_ok=True)
    with open("models/eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved eval metrics to models/eval_metrics.json")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Macro F1 : {metrics['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
