from dataclasses import dataclass
from re import S
from typing import Any
import jax.numpy as np
import numpy as onp
from jax import lax, ops, jit, random, vmap
from jax.ops import index, index_add, index_update






##############################################################





## 2d utils






#System(step_fn, steps, init_fns, program_fns).enum(program=canonical,init='random')


###############################################################




def enum_k1r1_programs():
    asdf

@dataclass
class CA_General_k1r1:
    params: Any
    config: Any

    def init_fns(self):
        return {
            'random':'random',
            'ascending':'ascending',
            'centered_ascending': lambda x: centered_init_k2_d1(self.config['width'], x)
        }

    def program_fns(self):
        return {
            'canonical': lambda x: integer_digits(x,2,8)
        }

    def enum(self):
        evolution_fn = eca_evolution
        program_fn_str = self.config.get('program_fn','canonical')
        program_fn = self.program_fns()[program_fn_str]
        num_programs = 5
        init_fn_str = self.config.get('init_fn','centered_ascending')
        init_fn = self.init_fns()[init_fn_str]
        # init_fn = partial(init_fn, config['width'])
        num_inits = 3
        e = Enum([
            ('program', program_fn, num_programs),
            ('init', init_fn, num_inits)
            ], None)
        
        return e.map("result", lambda x: evolution_fn(x['program'], x['init'], self.config['steps']))     


@dataclass
class CA_Totalistic_2d_k2r1:
    params: Any #sounds too complicated. wtf is this. 
    config: Any

    def init_fns(self):
        return {
            'random':'random',
            'ascending':'ascending',
            'centered_ascending': lambda x: d2_simple_init(self.config['width'], x+1, 2) 
        }

    def program_fns(self):
        return {
            'canonical': lambda x: totalistic_d2n9_bitcode(x,2)
        }

    def enum(self):
        evolution_fn = totalistic_2d_k2r1_evolution
        program_fn_str = self.config.get('program_fn','canonical')
        program_fn = self.program_fns()[program_fn_str]
        num_programs = 5
        init_fn_str = self.config.get('init_fn','centered_ascending')
        init_fn = self.init_fns()[init_fn_str]
        # init_fn = partial(init_fn, config['width'])
        num_inits = 3
        e = Enum([
            ('program', program_fn, num_programs),
            ('init', init_fn, num_inits)
            ], None)
        
        return e.map("result", lambda x: evolution_fn(x['program'], x['init'], self.config['steps']))    




#print(CA_General_k1r1({},{}).enum())   
#e = CA_General_k1r1({},{'width':5, 'steps':4}).enum()
#print(e.get(np.array([[0],[1],[2],[3],[4],[5]])))        

# print(CA_Totalistic_2d_k2r1({},{}).enum())   


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



     

#print(integer_digits(np.array([3],dtype=np.int32),2,8))

#print(d2_simple_init(9, np.arange(10,dtype=np.int32),2))

#print(index_update(np.zeros([10,10],dtype=np.int32), index[r2_inverse(np.arange(100, dtype=np.int32))], np.arange(100, dtype=np.int32)))

e = CA_Totalistic_2d_k2r1({},{'width':5, 'steps':4}).enum()
#print(e.sub_enums(np.array([[0],[1],[2],[3],[4],[5]])))    
#

#print(e.apply(e.sub_enums(np.array([[0],[1],[2],[3],[4],[5]]))))    
print(e.get(np.array([[1]]))) 