from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from jax.ops import index, index_add, index_update
from common import integer_digits_fn, iterations, iterations_2
from functools import partial

eca_bitcode = integer_digits_fn(2,8)

def eca_rulecase(left, center, right):
    return 7 - (right + 2 * (center + 2 * left))

def eca_step_fn(bitcode, init):
    wrapped = np.pad(init, [[1,1]], mode='wrap')
    left = wrapped[:-2]
    right = wrapped[2:]
    center = wrapped[1:-1]
    return np.take_along_axis(bitcode, eca_rulecase(left, center, right), 0)  

#def run_eca(bitcode, init, steps):
#    return iterations(eca_step_fn, bitcode, init, steps)




def run_eca2(bitcode, init, steps):
    return iterations_2(partial(eca_step_fn, bitcode), init, steps)

