from __future__ import annotations
import argparse
import time
import jax
from .solver import SolverConfig, run_simulation
from .diagnostics import norm, energy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nx", type=int, default=256)
    p.add_argument("--ny", type=int, default=256)
    p.add_argument("--lx", type=float, default=20.0)
    p.add_argument("--ly", type=float, default=20.0)
    p.add_argument("--g", type=float, default=100.0)
    p.add_argument("--dt", type=float, default=5e-4)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args()

    cfg = SolverConfig(
        nx=args.nx, ny=args.ny, lx=args.lx, ly=args.ly, g=args.g, dt=args.dt, steps=args.steps
    )

    t0 = time.perf_counter()
    out = run_simulation(cfg)
    psi = out["psi"]
    psi.block_until_ready()
    t1 = time.perf_counter()

    n = float(norm(psi, out["dx"], out["dy"]))
    e = float(energy(psi, out["V"], cfg.g, out["dx"], out["dy"]))

    print(f"backend={out['backend']}")
    print(f"devices={jax.devices()}")
    print(f"norm={n:.8f}")
    print(f"energy={e:.8f}")
    print(f"elapsed_s={t1-t0:.6f}")

    if args.benchmark:
        points = cfg.nx * cfg.ny * cfg.steps
        print(f"throughput_gridpoints_per_s={points/(t1-t0):.3e}")


if __name__ == "__main__":
    main()
