# cnn-vs-mlp
## Overview
## Setup
Requires Python 3.11 or newer.

```bash
git clone https://github.com/jeenhan/cnn-vs-mlp.git
cd cnn-vs-mlp
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
#### 1. Download Datasets
Start `python` from the repo root:

```bash
>>> from src.data import load
>>> x_train, y_train, x_test, y_test = load("mnist")
```
Check the shapes and the first 10 labels:
```bash
>>> x_train.shape, y_train.shape
((60000, 28, 28), (60000,))
>>> x_test.shape, y_test.shape
((10000, 28, 28), (10000,))
>>> y_test[:10]
array([7, 2, 1, 0, 4, 1, 4, 9, 5, 9], dtype=int32)
```

#### 2. Read Images and Verify the Pairing
Read images and labels and check visually if each image is correctly paired to its label.
```bash
cnn-vs-mlp % python -m scripts.show_samples
```

## Data
## Reproducing Results
## Architectural Differences from LeNet-5
## Contributions
