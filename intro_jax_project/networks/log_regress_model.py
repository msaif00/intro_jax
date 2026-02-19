import jax
from jax import random
import jax.numpy as jnp

key = random.key(0)

def sigmoid(x):
    return 0.5 * (jnp.tanh(x/2) + 1)

def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W)+b) #linear predict

# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                    [0.88, -1.08, 0.15],
                    [0.52, 0.06, -1.30],
                    [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])

#loss neg. log-likelihood
def loss(W, b):
    pred = predict(W,b, inputs)
    label_prob = pred * targets + (1-pred) * (1-targets)
    return jnp.sum(jnp.log(label_prob))

if __name__ == "__main__":
    # Initialize random model coefficients
    key, W_key, b_key = jax.random.split(key, 3)
    W = jax.random.normal(W_key, (3,))
    b = jax.random.normal(b_key, ())
    # Differentiate `loss` with respect to the first positional argument:
    W_grad = jax.grad(loss, argnums=0)(W, b)
    print(f'{W_grad=}')