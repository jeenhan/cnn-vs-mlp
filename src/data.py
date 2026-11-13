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

# Download the 4 dataset files into data/<name>/ and skip we already have
def download(name: str) -> None:
    folder = os.path.join(DATA_DIR, name)
    os.makedirs(folder, exist_ok=True)

    for filename in FILES.values():
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            print("Already have", path)
        else:
            print("Downloading", path)
            urllib.request.urlretrieve(URLS[name] + filename, path)

# Read a raw image into a float32 array of shape (N, 28, 28), values 0 to 1
def read_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    pixels = np.frombuffer(raw, dtype=np.uint8, offset=16) # (47040000,)
    images = pixels.reshape(-1, 28, 28) # Cut into 60000 pieces of 784
    return images.astype(np.float32) / 255.0

# Read a raw label into an int32 array of shape (N,), values 0 to 9
def read_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8)
    return labels.astype(np.int32)

# Return (x_train, y_train, x_test, y_test) for "mnist" or "fashion"
def load(name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download(name)
    folder = os.path.join(DATA_DIR, name)

    x_train = read_images(os.path.join(folder, FILES["x_train"]))
    y_train = read_labels(os.path.join(folder, FILES["y_train"]))
    x_test = read_images(os.path.join(folder, FILES["x_test"]))
    y_test = read_labels(os.path.join(folder, FILES["y_test"]))

    return x_train, y_train, x_test, y_test

# Split off n_val examples
def split_validation(
    x: np.ndarray,
    y: np.ndarray,
    n_val: int = 5000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x))
    valid_idx = idx[:n_val] # array([37141, 17573, 34459, ..., 45305, 10302, 28162], shape=(5000,))
    train_idx = idx[n_val:] # array([52660, 53583, 28430, ..., 54261,  6803, 49912], shape=(55000,)
    return x[train_idx], y[train_idx], x[valid_idx], y[valid_idx]
