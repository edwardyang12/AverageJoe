"""Uniform KL anchor, equivalent to plain entropy regularization."""

import jax.numpy as jnp


def uniform_magnet(obs, mask):
    """Give every action slot, including pass, the same probability.

    The policy applies its own legality mask. With this fixed uniform
    anchor, KL(policy || magnet) = -H(policy) + log(9 * H * W), so the
    regularizer has exactly the plain-entropy gradient.
    """
    h, w = mask.shape[:2]
    n_actions = 9 * h * w
    return jnp.full((n_actions,), 1.0 / n_actions, dtype=jnp.float32)


def get_magnet_fn(name):
    """Resolve the supported magnet without silently accepting old priors."""
    if name != "uniform":
        raise ValueError(f"Unknown magnet '{name}'. Available: uniform")
    return uniform_magnet
