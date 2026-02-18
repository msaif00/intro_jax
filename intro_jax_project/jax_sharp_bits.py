from functools import partial

import numpy as np
from jax import jit
from jax import lax
from jax import random
import jax
import jax.numpy as jnp
from jax import make_jaxpr
import time

"""
PURE Functions: JAX transformations and compilation are designed to work only on 
Python functions that are functionally pure: all the input data is passed through the function
parameters, all the results are output through the function results.

ITERATOR: Not recommended to use iterator with JAX because it introduces a python obj., that has a
state in order to retrieve the next element.

Asynchronous dispatch: Returns control to the user before the hardware has actually finished the math.
"""
def impure_print_side_effect(x):
    print('Executing function')
    return x

g=0
def impure_uses_globals(x):
    return x+g

g=0
def impure_saves_global(x):
    global g
    g=x
    return x

def pure_uses_internal_states(x):
    state = dict(even=0, odd=0)
    for i in range(10):
        state['even' if i%2==0 else 'odd'] += x
    return state['even'] + state['odd']

@jax.jit
def func11(arr, extra):
    def body(carry, x):
        return (carry + x + extra, carry)
    return lax.scan(body, 0., arr)

def func_standard(arr, extra):
    def body(carry, x):
        return (carry + x + extra, carry)
    return lax.scan(body, 0, arr)

@jax.jit
def func_jitted(arr, extra):
    def body(carry, x):
        return (carry + x + extra, carry)
    return lax.scan(body, 0, arr)


###jax.jit with class methods

class CustomClass:
    def __init__(self, x, mul):
        self.x = x
        self.mul = mul
    def calc(self, y):
        return _calc(self.mul, self.x, y)

@partial(jit, static_argnums=0)
def _calc(mul, x, y):
    if mul:
        return x * y
    return y

if __name__=="__main__":
    # print("First call: ", jit(impure_saves_global)(4.))
    # print("Second call: ", jit(impure_uses_globals)(5.)) #uses cached val at 0 not 10
    # print("Third call, different type: ", jit(impure_uses_globals)(jnp.array( [4.] )))
    # print("Saved global ", g)
    # print(jit(pure_uses_internal_states)(5.))
    # lax.fori_loop
    # array = jnp.arange(10)
    # print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0))
    # iterator = iter(range(10))
    # print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0))
    # make_jaxpr(func11)(jnp.arange(16), 5.)
    # #lax.scan
    # # Example Usage:
    # array_input = jnp.array([1.0, 2.0, 3.0])
    # extra_val = 10.0
    # final_carry, running_history = func11(array_input, extra_val)
    # prin(f"History: {running_history}")

    # size = 100000
    # arr = jnp.ones(size)
    # extra = 5.0
    #
    # #warm up
    # _ = func_jitted(arr, extra)
    #
    # start = time.time()
    # res1 = func_standard(arr, extra)[1].block_until_ready()
    # print(f"Standard time: {(time.time() - start):.6f}s")
    #
    # start = time.time()
    # res2 = func_jitted(arr, extra)[1].block_until_ready()
    # print(f"JIT time:      {(time.time() - start):.6f}s")

    ##Array updates with .at

    # jax_array = jnp.zeros((3,3), dtype=jnp.float32)
    # updated_array = jax_array.at[1, :].set(1.0)
    # print("updated array:\n", updated_array)
    # print("original array unchanged:\n", jax_array)
    c = CustomClass(2, True)
    print(c.calc(3))
