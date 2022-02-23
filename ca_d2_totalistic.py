from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from jax.ops import index, index_add, index_update
from common import integer_digits, random_init, run_2d

def totalistic_d2n9_bitcode_fn(k):
    return integer_digits(k, (9 * (k - 1) + 1))

def totalistic_d2n9_step_fn(bitcode, init):
    wrapped = np.pad(init, [[1,1],[1,1]], mode='wrap')
    total = lax.reduce_window(
      operand=wrapped, 
      init_value=0, 
      computation=np.add, 
      window_dimensions=(3, 3), 
      window_strides=(1, 1), 
      padding='SAME')
    total = total[1:-1,1:-1]
    ## wtf is below?
    result = np.reshape(
      np.take_along_axis(
        bitcode,
        np.reshape(total, (bitcode.shape[0],-1)), 
        1),
      total.shape)
    return result