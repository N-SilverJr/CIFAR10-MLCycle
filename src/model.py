"""CNN architecture, training, and retraining logic for CIFAR-10.

Shared by the notebook and the API.
"""
import os
from datetime import datetime

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks

from src.preprocessing import preprocess_raw_arrays

TEST_X_PATH = "data/test/X_test.npy"
TEST_Y_PATH = "data/test/y_test.npy"
NUM_CLASSES = 10


def build_enhanced_cnn():
    """Enhanced CNN for CIFAR-10 (32x32 RGB).

    Architecture chosen after notebook experiments:
    - 3 convolutional blocks with BatchNorm + Dropout
    - GlobalAveragePooling instead of Flatten to reduce parameters
    - Dense head with Dropout
    ~1.2M parameters – still lightweight enough for Docker free-tier scaling.
    """
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name="cifar10_enhanced_cnn")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=30, batch_size=64):
    """Train with early stopping and learning-rate reduction on plateau."""
    training_callbacks = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=training_callbacks,
        verbose=2,
    )
    return history.history


def retrain_model(model_path, X_new, y_new, epochs=5, batch_size=32, tolerance=0.01):
    """Fine-tune a saved model on newly collected data, gated by a safety check.

    The retrained model is only promoted (saved over model_path) if its accuracy
    on the held-out test set stays within `tolerance` of the pre-retrain baseline.
    Default tolerance is 1 percentage point (CIFAR-10 is harder than MNIST).

    Returns a dict with promoted, baseline_accuracy, new_accuracy, accuracy_change,
    samples_used, epochs_run, reason.
    """
    model = keras.models.load_model(model_path)

    X_test_raw = np.load(TEST_X_PATH)
    y_test_raw = np.load(TEST_Y_PATH)
    X_test, y_test = preprocess_raw_arrays(X_test_raw, y_test_raw)

    _, baseline_accuracy = model.evaluate(X_test, y_test, verbose=0)

    # Recompile – optimizer state is not restored from a plain .h5 save
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # lower LR for fine-tune
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_new, y_new,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )

    _, new_accuracy = model.evaluate(X_test, y_test, verbose=0)

    baseline_accuracy = float(baseline_accuracy)
    new_accuracy = float(new_accuracy)
    accuracy_change = new_accuracy - baseline_accuracy
    promoted = new_accuracy >= (baseline_accuracy - tolerance)

    if promoted:
        model.save(model_path)
        reason = (
            f"Retrained model accuracy ({new_accuracy:.4f}) is within tolerance "
            f"of baseline ({baseline_accuracy:.4f}). Model promoted."
        )
    else:
        reason = (
            f"Retrained model accuracy ({new_accuracy:.4f}) regressed beyond "
            f"{tolerance * 100:.1f}% tolerance from baseline ({baseline_accuracy:.4f}). "
            f"Original model kept."
        )

    return {
        "promoted": bool(promoted),
        "baseline_accuracy": baseline_accuracy,
        "new_accuracy": new_accuracy,
        "accuracy_change": accuracy_change,
        "samples_used": int(len(X_new)),
        "epochs_run": len(history.history["loss"]),
        "reason": reason,
    }


_model_info_cache = {"mtime": None, "params": None}


def get_model_info(model_path):
    """Return file size, last-modified timestamp and parameter count.

    Parameter count is cached and only recomputed when the file mtime changes.
    """
    if not os.path.exists(model_path):
        return {
            "exists": False,
            "size_bytes": None,
            "last_modified": None,
            "parameters": None,
        }

    size_bytes = os.path.getsize(model_path)
    mtime = os.path.getmtime(model_path)
    last_modified = datetime.fromtimestamp(mtime).isoformat()

    if _model_info_cache["mtime"] != mtime:
        model = keras.models.load_model(model_path)
        _model_info_cache["mtime"] = mtime
        _model_info_cache["params"] = model.count_params()

    return {
        "exists": True,
        "size_bytes": size_bytes,
        "last_modified": last_modified,
        "parameters": _model_info_cache["params"],
    }
