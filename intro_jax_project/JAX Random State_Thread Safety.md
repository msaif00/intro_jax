### Global State (Not Thread Safe)
In libraies like np, there is a hidden global state, that 
1. reads the current global state (a seed).
2. Calculates a random number.
3. Updates (mutates) that global state so that the next call gets a different number.

**The Problem**: If we are running two different threads at the same time, and both call
np.random.normal(), they both try to read and update that same global variable simulanteously.
This is a **race condition**. There is a chance that the same number will be generated at both threads,
or the state may become corrupted, then leading to unpredictable results, making it impossible to
reproduce the work exactly.

**JAX Solution**: JAX uses an explicit "key" (called PRNGKey).

```python
from jax import random
key = random.key(42)
val = random.normal(key)
```
This key is **immutable**, the "random.normal(key" completes the calculation using the key and
returns a value. There is **no shared resource** among different threads and since the state is local,
the output depends only on that input, it is a **deterministic** calculation.

