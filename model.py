# next: put in lib, upload to github, & pull down into colab
# next: similar training loop as below, but batch the rules

from os import stat_result
from ca_eca import CellularAutomatonK2R1, Enum
import jax.numpy as np
import flax
from flax.core import freeze, unfreeze
from flax import linen as nn
from jax import lax, random, numpy as jnp
import optax
from flax.training import train_state
import jax
from jax import vmap, jit
import time

def sigmoid_cross_entropy_with_logits(x, z):
  return jnp.mean(jnp.maximum(x, 0) - x * z + jnp.log1p(jnp.exp(-jnp.abs(x))))

# https://github.com/google/jax/issues/7171


def create_train_state(rng):
  """Creates initial `TrainState`."""
  model = nn.Dense(features=1)
  sample_input = jnp.ones([8, 3])
  params = model.init(rng, sample_input)['params']
  learning_rate=0.1
  momentum=0.01
  tx = optax.sgd(learning_rate, momentum)
  return train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)

@jit
def apply_model(state, images, labels):
  """Computes gradients, loss and accuracy for a single batch."""
  def loss_fn(params):
    logits = nn.Dense(features=1).apply({'params': params}, images)
    loss = sigmoid_cross_entropy_with_logits(logits, labels)
    return loss, logits

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(state.params)
  accuracy = jnp.sum(jnp.round(jax.nn.sigmoid(logits)) == labels)
  return grads, loss, accuracy

@jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)


ca = CellularAutomatonK2R1(width=3,steps=1)
r = ca.enum_rules()
s = ca.enum_states()
rb = r.batch(256)[0]
sb = s.batch(8)[0]
evolutions = ca.evolve_batch(rb, sb)
evolutions_inputs = evolutions[:,:,0] * 1.0
evolutions_labels = evolutions[:,:,1,0] * 1.0

evolutions_inputs = evolutions_inputs[0:256]
evolutions_labels = jnp.expand_dims(evolutions_labels[0:256], (-1,))

rng, init_rng = random.split(random.PRNGKey(0))
init_rng = jnp.tile(init_rng, (256,1))
state = vmap(create_train_state)(init_rng)

epoch_loss = []
epoch_accuracy = []
t0 = time.time()
for i in range(1000):
  print(i)
  grads, loss, accuracy = vmap(apply_model)(state, evolutions_inputs, evolutions_labels)
  state = vmap(update_model)(state, grads)
  epoch_loss.append(loss)
  print(jnp.sum(accuracy))
t1 = time.time()
print(t1-t0)  
#print("loss")
#print(epoch_loss)  




# # science Q: can we tell the difference between boolean functions & CA?
# # can take CAs from degenerate case of just the rule case, and then add width to get 
# # boolean functions. Where do those lie in the space of all boolean functions of that width?

# effect of symmetry in totalistic vs general rules?

# Jax library for modelling emergent computation, in the spirit of NKS and AIT
# 