## Standard libraries
import os
import math
import numpy as np
import time

## Imports for plotting
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import seaborn as sns
sns.set()
import jax
import jax.numpy as jnp
print("Using jax", jax.__version__)
## Progress bar
from tqdm.auto import tqdm

a = jnp.arange(10)
print(a.device)