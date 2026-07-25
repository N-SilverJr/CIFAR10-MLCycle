"""
Locust load-test script for the CIFAR-10 prediction endpoint.
Uses a small synthetic RGB image so the test does not depend on real data files.
"""
import io
import random
from locust import HttpUser, task, between
from PIL import Image
import numpy as np


def make_dummy_image_bytes():
    """Create a random 32x32 RGB PNG in memory."""
    arr = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()


class CIFAR10User(HttpUser):
    wait_time = between(0.5, 1.5)

    def on_start(self):
        self.image_bytes = make_dummy_image_bytes()

    @task
    def predict(self):
        files = {"file": ("test.png", self.image_bytes, "image/png")}
        with self.client.post("/predict", files=files, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "predicted_class" in data or "predicted_index" in data:
                    response.success()
                else:
                    response.failure("Missing prediction fields")
            else:
                response.failure(f"Status {response.status_code}")
