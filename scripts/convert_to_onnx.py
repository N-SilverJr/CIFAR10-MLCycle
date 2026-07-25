"""
Convert cifar10_classifier.h5 to cifar10_classifier.onnx for lightweight cloud deployment.
Run from project root after training and saving a SavedModel:
    python scripts/convert_to_onnx.py
"""
import os
import subprocess
import sys
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

model = tf.keras.models.load_model("models/cifar10_classifier.h5")

output_path = "models/cifar10_classifier.onnx"
savedmodel_path = "models/cifar10_classifier_savedmodel"

# Prefer SavedModel path for reliable tf2onnx conversion with recent Keras
if not os.path.exists(savedmodel_path):
    model.export(savedmodel_path)

subprocess.run(
    [
        sys.executable, "-m", "tf2onnx.convert",
        "--saved-model", savedmodel_path,
        "--output", output_path,
        "--opset", "13",
    ],
    check=True,
)

# Verify predictions match
test_input = np.random.rand(1, 32, 32, 3).astype(np.float32)
tf_pred = model.predict(test_input, verbose=0)

import onnxruntime as ort
session = ort.InferenceSession(output_path)
input_name = session.get_inputs()[0].name
onnx_pred = session.run(None, {input_name: test_input})[0]

max_diff = np.max(np.abs(tf_pred - onnx_pred))
print(f"Saved ONNX model to {output_path}")
print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
print(f"  Max prediction difference between TF and ONNX: {max_diff:.8f}")
if max_diff < 1e-4:
    print("  Verification PASSED.")
else:
    print("  WARNING: predictions differ more than expected.")
