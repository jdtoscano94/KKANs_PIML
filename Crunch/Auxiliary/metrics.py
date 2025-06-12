import time
import jax
from jax import vmap
import jax.numpy as jnp
import optax

def relative_l2(u, u_gt):
    return jnp.linalg.norm(u-u_gt) / jnp.linalg.norm(u_gt)

def relative_l2m(u, u_gt):
    return jnp.linalg.norm(u-u_gt,2) / jnp.linalg.norm(u_gt,2)


def relative_l2_2(u, u_gt):
    return jnp.linalg.norm(u.flatten()-u_gt.flatten(),2) / jnp.linalg.norm(u_gt.flatten(),2)