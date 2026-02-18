from jax import random
import numpy as np


"""
Numpy supports a method of pseduo random number generation that is based 
on a global state. JAX, however, tracks state explicity via a random 'key'.
The key is effectively a stand-in for np's hidden state object, but we pass
it explicity to jax.random functions.
JAX requires:
    reproducible,

    parallelizable,

    vectorisable.
"""


if __name__ == "__main__":
    ## Never re-use keys (unless you want identical outputs).
    # key = random.key(43)
    # print(key)
    ## np evaluates cocde in the order defined by the
    ##Python interpreter, however, with JAX, for efficient execution we want the JIT
    ## compiler to be free to reorder, elide, and fuse funcs.
    np.random.seed(0)
    def bar(): return np.random.uniform()
    def baz(): return np.random.uniform()
    def foo(): return bar() + 2 * baz()
    print(foo())
