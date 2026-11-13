"""Download and load MNIST and Fashion-MNIST."""
import os
import urllib.request

import gzip
import numpy as np

URLS = {
    "mnist": "http://storage.googleapis.com/cvdf-datasets/mnist/", # + filename
    "fashion": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
}

# Each dataset has 4 files (same names for both)
FILES = {
    "x_train": "train-images-idx3-ubyte.gz", # 60k training images
    "y_train": "train-labels-idx1-ubyte.gz", # 60k training labels (0 to 9)
    "x_test": "t10k-images-idx3-ubyte.gz", # 10k test images
    "y_test": "t10k-labels-idx1-ubyte.gz", # 10k test labels
}

# The data/ folder is at the top of the repo
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

CLASS_NAMES = {
    "mnist": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    "fashion": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
}

# Download the 4 files for "mnist" or "fashion" into data/<name>/
def download(name: str) -> None:
    folder = os.path.join("data", name)
    os.makedirs(folder, exist_ok=True)

    for filename in FILES.values():
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            print("Already have", path)
        else:
            print("Downloading", path)
            urllib.request.urlretrieve(URLS[name] + filename, path)

# Reshape and convert raw data
def read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16)
    images = pixels.reshape(-1, 28, 28)
    return images.astype(np.float32) / 255.0

def read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return labels.astype(np.int32)

def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download(name)
    folder = os.path.join("data", name)

    x_train = read_images(os.path.join(folder, FILES["x_train"]))
    y_train = read_labels(os.path.join(folder, FILES["y_train"]))
    x_test = read_images(os.path.join(folder, FILES["x_test"]))
    y_test = read_labels(os.path.join(folder, FILES["y_test"]))

    return x_train, y_train, x_test, y_test
