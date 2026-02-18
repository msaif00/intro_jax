import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random

### Array Creation
x = jnp.arange(5)
print(isinstance(x, jax.Array))

### Array Devices and Sharding
print(x.devices())
print(x.sharding)

###JIT Compilation
## all arrays must have static shapes, so op-by-op cannot be
##jitted

###Asynchronous dispatch


### jax.grad

###jax.vmap
key = random.key(1701)
key1, key2 = random.split(key)
mat = random.normal(key1, (150,100))
batched_x = random.normal(key2,(10,100))

def apply_matrix(x):
    return jnp.dot(mat,x)

def naively_batched_apply_matrix(v_batched):
  return jnp.stack([apply_matrix(v) for v in v_batched])

print('Naively batched')
naively_batched_apply_matrix(batched_x).block_until_ready()