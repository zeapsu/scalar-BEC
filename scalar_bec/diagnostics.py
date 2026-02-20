from __future__ import annotations
import jax.numpy as jnp


def l2_error(psi, psi_ref, dx, dy):
    return jnp.sqrt(jnp.sum(jnp.abs(psi - psi_ref) ** 2) * dx * dy)


def norm(psi, dx, dy):
    return jnp.sum(jnp.abs(psi) ** 2) * dx * dy


def energy(psi, V, g, dx, dy):
    gradx = jnp.gradient(psi, dx, axis=0)
    grady = jnp.gradient(psi, dy, axis=1)
    kinetic = 0.5 * (jnp.abs(gradx) ** 2 + jnp.abs(grady) ** 2)
    potential = V * jnp.abs(psi) ** 2
    interaction = 0.5 * g * jnp.abs(psi) ** 4
    return jnp.real(jnp.sum((kinetic + potential + interaction) * dx * dy))
