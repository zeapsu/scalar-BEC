from __future__ import annotations
from dataclasses import dataclass
from functools import partial
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class SolverConfig:
    nx: int = 256
    ny: int = 256
    lx: float = 20.0
    ly: float = 20.0
    g: float = 100.0
    dt: float = 5e-4
    steps: int = 1000


def make_grid(cfg: SolverConfig):
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    x = (jnp.arange(cfg.nx) - cfg.nx // 2) * dx
    y = (jnp.arange(cfg.ny) - cfg.ny // 2) * dy
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    return x, y, X, Y, dx, dy


def harmonic_potential(X, Y, omega=1.0):
    return 0.5 * omega**2 * (X**2 + Y**2)


def kspace_operators(cfg: SolverConfig):
    kx = 2.0 * jnp.pi * jnp.fft.fftfreq(cfg.nx, d=cfg.lx / cfg.nx)
    ky = 2.0 * jnp.pi * jnp.fft.fftfreq(cfg.ny, d=cfg.ly / cfg.ny)
    KX, KY = jnp.meshgrid(kx, ky, indexing="ij")
    k2 = KX**2 + KY**2
    kinetic_phase = jnp.exp(-0.5j * cfg.dt * k2)
    return kinetic_phase


def initial_gaussian(X, Y, sigma=1.0):
    psi = jnp.exp(-(X**2 + Y**2) / (2.0 * sigma**2)).astype(jnp.complex64)
    return psi / jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2))


def normalize(psi, dx, dy):
    norm = jnp.sqrt(jnp.sum(jnp.abs(psi) ** 2) * dx * dy)
    return psi / norm


def nonlinear_phase(psi, V, g, dt_half):
    return jnp.exp(-1j * dt_half * (V + g * jnp.abs(psi) ** 2))


@partial(jax.jit, static_argnames=("g",))
def strang_step(psi, V, kinetic_phase, dt, g):
    half = 0.5 * dt
    phase1 = jnp.exp(-1j * half * (V + g * jnp.abs(psi) ** 2))
    psi = phase1 * psi

    psi_k = jnp.fft.fft2(psi)
    psi_k = kinetic_phase * psi_k
    psi = jnp.fft.ifft2(psi_k)

    phase2 = jnp.exp(-1j * half * (V + g * jnp.abs(psi) ** 2))
    psi = phase2 * psi
    return psi


def run_simulation(cfg: SolverConfig):
    x, y, X, Y, dx, dy = make_grid(cfg)
    V = harmonic_potential(X, Y)
    psi = normalize(initial_gaussian(X, Y), dx, dy)
    kinetic_phase = kspace_operators(cfg)

    def body_fun(_, state):
        return strang_step(state, V, kinetic_phase, cfg.dt, cfg.g)

    psi = jax.lax.fori_loop(0, cfg.steps, body_fun, psi)
    psi = normalize(psi, dx, dy)
    return {
        "psi": psi,
        "x": x,
        "y": y,
        "dx": dx,
        "dy": dy,
        "V": V,
        "config": cfg,
        "backend": jax.default_backend(),
    }
