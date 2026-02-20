import jax
import jax.numpy as jnp
from jax import random

from mlp import loss_fn, params

## Dummy Data
key = random.PRNGKey(42)
k1,k2 = random.split(key)
x_data = random.normal(k1, (1000, 784))
y_data = random.randint(k2, (1000, 10), 0, 2)

learning_rate = 0.001
steps= 1000

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    new_params =  jax.tree_util.tree_map(
        lambda p,g: p-learning_rate * g, params, grads
    )
    return new_params

for i in range(steps):
    params = update(params, x_data, y_data)

    if i % 20 == 0:
        current_loss = loss_fn(params, x_data, y_data)
        print(f"Step {i}, Loss {current_loss:.2f}")