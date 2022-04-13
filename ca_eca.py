from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
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
    return iterations_3(partial(eca_step_fn, bitcode), init, steps)


def iterations_4(step_fn, init, steps):
    def _step_fn(state, _):
        state_ = step_fn(state)
        return state_,state_
    # add bit to join the init onto the stacked state    
    _, evolution = lax.scan(_step_fn, init, None, steps)
    print(evolution)
    print(np.expand_dims(init, 0))
    evolution = np.concatenate([np.expand_dims(init, 0), evolution])
    return evolution

class CellularAutomatonK2R1:
    def __init__(this, width, steps):
        this.width = width
        this.steps = steps
        this.max_addressable_width = 30
    
    def evolve(this, rule, state):
        # assume single rule, single state
        bitcode = eca_bitcode(rule)
        return iterations_4(partial(eca_step_fn, bitcode), state, this.steps)
    def evolve_batch(this, rules, states):
        # do cartesian product. assumes batches
        return vmap(lambda r: vmap(lambda s: this.evolve(r,s))(states))(rules)
        
    # return non-pre-batched enums
    def enum_rules(this):
        return Enum(lambda x: x, 256)
    def enum_rules_random(this):
        pass
    def enum_states(this):
        decoder = integer_digits_fn(2,this.width)
        return Enum(decoder, 2**min(this.width, this.max_addressable_width))
    def enum_states_random(this):
        decoder = random_int_fn(this.width,2)
        return Enum(decoder, 2**min(this.width, this.max_addressable_width))

def placeholder():
# case 1: model per rule, iterate 1 init at a time    
# first iterate through rules
# how do we set up the actual modelling code -- needs state too right?
    for canonical_rule, indexed_rule in ca.enum_rules().batch(256):
        # for each rule, iterate thru states
        for states, state_index in ca.enum_states_random.batch(1):
            # run all rules for all states
            evolutions = ca.evolve(canonical_rule, states)
            vmap(model_train)(evolutions)
        # yield model_per_rule

    # first iterate through states
    for states, states_index in ca.enum_states(offset=100).batch(1):
        # for each state, iterate through rules
        for canonical_rule, indexed_rule in ca.enum_rules(batch_size=256): # single batch
            # train classifier that potentially sees different rules per state
            evolutions = ca.evolve(canonical_rule, states)
            # be more specific about data vs labels
            model_train(evolutions)

# https://github.com/google/flax/blob/main/examples/mnist/train.py


class Enum:
    def __init__(self, f, size):
        self.f = f
        self.size = size
    
    def batch(self, batch_size):
        return Enum(lambda x: self[x*batch_size:(x+1)*batch_size], self.size / batch_size) 

    # is this jittable?
    def __getitem__(self, key):
        if type(key) == int:
            if 0 <= key < self.size:
                return self.f(key)
            else:
                raise IndexError()  
        elif type(key) == slice:
            return vmap(self.f)(np.arange(key.start, key.stop))



def random_int_fn(width, k):
    def _random(i):
        return random.randint(np.array([0,i], dtype=np.uint32), [width], 0, k, dtype=np.int32)
    return _random

