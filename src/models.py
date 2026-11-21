"""Parameter initialization and forward passes."""
import numpy as np

from src.layers import dense, init_dense, relu

# Weights and biases in (784 -> n_hidden -> 10)
def init_mlp(seed=0, n_hidden=128):
    rng = np.random.default_rng(seed)
    W1, b1 = init_dense(rng, 784, n_hidden)
    W2, b2 = init_dense(rng, n_hidden, 10)
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

# Images (batch, 28, 28) -> scores (batch, 10)
def mlp_forward(params, x):
    """
    x: (64, 28, 28)
    Returns (64, 10); 10 scores per image (logits)
    """
    x = x.reshape(x.shape[0], -1) # (64, 28, 28) -> (64, 784)
    h = relu(dense(params["W1"], params["b1"], x)) # (64, 784) -> (64, 128)
    return dense(params["W2"], params["b2"], h) # (64, 128) -> (64, 10)
