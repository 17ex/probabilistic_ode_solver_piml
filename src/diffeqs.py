import jax
import jax.numpy as jnp
from jaxtyping import Array
from functools import partial


@partial(jax.jit, static_argnames=["args"])
def linear_vf(t, y, args):
    alpha = args[0]
    return alpha * y


@partial(jax.jit, static_argnames=["args"])
def lotka_volterra_vf(t, y, args) -> Array:
    prey, predator = y[..., 0], y[..., 1]
    alpha, beta, gamma, delta = args
    d_prey: Array = alpha * prey - beta * prey * predator
    d_predator: Array = -gamma * predator + delta * prey * predator
    return jnp.array([d_prey, d_predator])
