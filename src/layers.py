"""Building bocks written with JAX primitives."""
import jax
import jax.numpy as jnp
import numpy as np

# Initialize weights and biases
def init_dense(rng, n_in: int, n_out: int) -> tuple[jax.Array, jax.Array]:
    W = rng.normal(size=(n_in, n_out)) * np.sqrt(2.0 / n_in)
    b = np.zeros(n_out)
    return jnp.array(W), jnp.array(b)

# Fully connected layer
def dense(W: jax.Array, b: jax.Array, x: jax.Array) -> jax.Array:
    """
    x: (batch, n_in)
    W: (n_in, n_out)
    Returns (batch, n_out)
    """
    return x @ W + b

def relu(x: jax.Array) -> jax.Array:
    return jnp.maximum(0, x)

# Scores -> probabilities
def softmax(scores: jax.Array) -> jax.Array:
    """
    scores: (64, 10)
    Returns (64, 10); Each row sums to 1.
    """
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    e = jnp.exp(scores)
    return e / jnp.sum(e, axis=-1, keepdims=True)
