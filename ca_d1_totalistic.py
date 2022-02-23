import common
from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from jax.ops import index, index_add, index_update
from common import integer_digits_fn, iterations, iterations_2
from functools import partial

def totalistic_ruletable_numentries(k, r):
    # for each cell, maximum value is k-1
    # have 2*r+1 cells in the neighborhood
    # maximum possible totaled value is num_cells * max_value)
    # have a rulecase for values [0, ..., maxvalue], aka maxvalue + 1
    return (2*r+1) * (k - 1) + 1

def totalistic_d1_bitcode(k, r):
    return common.integer_digits_fn(k, totalistic_ruletable_numentries(k, r))   

def totalistic_d1_r1_step_fn(bitcode, init):
    wrapped = np.pad(init, [[1,1]], mode='wrap')
    total = lax.reduce_window(
        wrapped,
        0,
        np.add,
        (3,),
        (1,),
        'SAME')
    total = total[1:-1]
    return np.take_along_axis(bitcode, 6 - total, 0)

#####

def outer_totalistic_rulecase(k, center, total):
  return k*total + center

def outer_totalistic_ruletable_numentries(k, r):
    # similar to totalistic case, but now have additional branching according to value of 
    # center cell. 
    return k * (2 * r * (k - 1) + 1)


def outer_totalistic_bitcode(k, r):
    return common.integer_digits_fn(k, outer_totalistic_ruletable_numentries(k, r))

def outer_totalistic_step_fn(bitcode, init):
    wrapped = np.pad(init, [[1,1]], mode='wrap')
    k=3
    center = wrapped[1:-1]
    total = lax.reduce_window(
      wrapped,
      0,
      np.add,
      (3,),
      (1,),
      'SAME')
    total = total[1:-1]
    total = total - center
    return np.take_along_axis(bitcode, 14 - outer_totalistic_rulecase(k, center, total), 0) 