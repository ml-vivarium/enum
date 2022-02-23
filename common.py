from dataclasses import dataclass
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from jax.ops import index, index_add, index_update
from functools import partial

def integer_digits_fn(base, n):
  def _integer_digits(i):
    def digits_body(carry, input_slice, base):
      n, r = np.divmod(carry,base)
      return n, r

    result = lax.scan(
      lambda a,b: digits_body(a ,b, base), 
      i, 
      np.zeros([n], np.int32))
 
    return np.flip(result[1])

  return _integer_digits

def iterations(step_fn, bitcode, init, steps):
    def _step_fn(state, _):
        r = step_fn(bitcode, state)
        return r,r
    # in the vmapped case, need to transpose the stacked states?    
    return lax.scan(_step_fn, init, None, steps)

def iterations_2(step_fn, init, steps):
    def _step_fn(state, _):
        state_ = step_fn(state)
        return state_,state_
    # add bit to join the init onto the stacked state    
    return lax.scan(_step_fn, init, None, steps)  

def evolve_fn(step_fn, bitcode_fn, state_fn):
    def evolve(rule_number, state_number, steps):
        return iterations_2(partial(step_fn, bitcode_fn(rule_number)), state_fn(state_number), steps)
    return evolve


# def integer_digits(numbers, base, n):
#   def digits_body(carry, input_slice, base):
#     n, r = np.divmod(carry,np.array([base]))
#     return n, r

#   result = lax.scan(
#     lambda a,b: digits_body(a,b, base), 
#     numbers, 
#     np.zeros([n, numbers.shape[0]], np.int32))
 
#   return np.flip(result[1].T,1)

################### 1D

def random_int_fn(width, k):
    def _random(i):
        return random.randint(np.array([0,i], dtype=np.uint32), [width], 0, k, dtype=np.int32)
    return _random


# def random_init(width,k,ids):
#   keys = np.stack(
#       [np.zeros([len(ids)],dtype=np.uint32), 
#        ids # note, may need to case to uint32
#        ]).T
#   return vmap(lambda key: random.randint(key, [width], 0, k, dtype=np.int32))(keys)  
    

def centered_init_k2_d1_fn(width):
    def _init(i):
        return np.roll(
            integer_digits_fn(2, width)(i),
            np.ceil(width/2).astype(np.int32))
    return _init         

# def centered_init_k2_d1(width,ids):
#     return np.roll(
#         integer_digits(ids, 2, width), 
#         np.ceil(width/2).astype(np.int32),
#         axis=1)  

def inits_asc_k2_d1(width, ids):
    return integer_digits(ids, 2, width)

def buffer_fn_1d(steps, width, batch_size):
    return np.zeros([steps, batch_size, width], dtype=np.int32)

def run_1d(step_fn, bitcode, init, steps):
  ## only good for 1D right now  
  def scan_fn(previous, current):
    x = step_fn(bitcode, previous)
    return (x, x)

  evolution = lax.scan(
      f=scan_fn,
      init=init,
      xs=buffer_fn_1d(steps,init.shape[-1],bitcode.shape[0])
  )[1]

  evolution = np.transpose(evolution, (1,0,2))
  evolution = np.concatenate([np.expand_dims(init, 1), evolution], 1)

  return evolution 

############# 2D

def buffer_fn_2d(steps, width, batch_size):
    return np.zeros([steps, batch_size, width, width], dtype=np.int32)  

def run_2d(step_fn, bitcode, init, steps):
  def scan_fn(previous, current):
    x = step_fn(bitcode, previous)
    return (x, x)

  evolution = lax.scan(
      f=scan_fn,
      init=init,
      xs=buffer_fn_2d(steps, init.shape[-1], bitcode.shape[0])
  )[1]

  evolution = np.transpose(evolution, (1,0,2, 3))
  evolution = np.concatenate([np.expand_dims(init, 1), evolution], 1)

  return evolution    


def d2_simple_init(size, n, b):
    digits = integer_digits(n,b,size*size)
    unpaired = r2_inverse(np.arange(size*size, dtype=np.int32))
    x = np.arange(n.shape[0], dtype=np.int32)
    i = unpaired
    print(i)
    def updater(x):
        return index_update(
            np.zeros([size,size],dtype=np.int32), 
            index[i],
            np.flip(x)
        )
    return vmap(updater)(digits)   

# https://arxiv.org/pdf/1706.04129.pdf
def r2(x, y): 
    return np.power(np.maximum(x, y), 2) + np.maximum(x, y) + x - y

def r2_inverse(z):
    m = np.floor(np.sqrt(z)).astype(np.int32)
    condition = z - np.power(m,2) < m
    branch_true = (z - np.power(m, 2), m)
    branch_false = (m, np.power(m, 2) + 2*m - z)
    return (
        np.where(condition, branch_true[0], branch_false[0]),
        np.where(condition, branch_true[1], branch_false[1])
    )       
