from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from common import run_2d
import ca_jax

def totalistic_d2n9_bitcode_fn(k):
    table_size = ca_jax.table_size_totalistic(k, ca_jax.moore_offsets(2, 1))
    return lambda rule: ca_jax.wolfram_rule_table(rule, k, table_size)

def totalistic_d2n9_step_fn(bitcode, init):
    offsets = ca_jax.moore_offsets(2, 1)
    weights = [1] * len(offsets)
    kernel = ca_jax.kernel_from_offsets(offsets, weights)
    return ca_jax.step_wrapped_convolution(init, bitcode, kernel)


def outer_totalistic_d2n9_bitcode_fn(k):
    table_size = ca_jax.table_size_outer_totalistic(k, ca_jax.moore_offsets(2, 1))
    return lambda rule: ca_jax.wolfram_rule_table(rule, k, table_size)


def outer_totalistic_d2n9_step_fn(bitcode, init, k=2):
    offsets = ca_jax.moore_offsets(2, 1)
    weights = [1 if offset == (0, 0) else k for offset in offsets]
    kernel = ca_jax.kernel_from_offsets(offsets, weights)
    return ca_jax.step_wrapped_convolution(init, bitcode, kernel)


def game_of_life_table():
    offsets = ca_jax.moore_offsets(2, 1)
    outputs = [0] * ca_jax.table_size_outer_totalistic(2, offsets)
    outputs[2 * 3 + 0] = 1
    outputs[2 * 2 + 1] = 1
    outputs[2 * 3 + 1] = 1
    return ca_jax.table_from_outputs(outputs)


def game_of_life_step_fn(init):
    return outer_totalistic_d2n9_step_fn(game_of_life_table(), init, k=2)
