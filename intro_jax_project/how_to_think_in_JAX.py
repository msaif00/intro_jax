import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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

