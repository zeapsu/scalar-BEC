from __future__ import annotations
import jax.numpy as jnp
import pandas as pd
from scalar_bec.solver import SolverConfig, run_simulation
from scalar_bec.diagnostics import l2_error


def resample_ref(ref, n):
    # periodic-style subsampling for fair spectral comparison
    stride = ref.shape[0] // n
    return ref[::stride, ::stride]


def grid_convergence():
    ref_cfg = SolverConfig(nx=512, ny=512, steps=800, dt=2.5e-4)
    ref = run_simulation(ref_cfg)["psi"]

    rows = []
    for n in [64, 128, 256]:
        cfg = SolverConfig(nx=n, ny=n, steps=800, dt=2.5e-4)
        out = run_simulation(cfg)
        psi_n = out["psi"]
        psi_ref_n = resample_ref(ref, n)
        err = float(l2_error(psi_n, psi_ref_n, out["dx"], out["dy"]))
        rows.append({"n": n, "l2_error_vs_ref": err})
        print(f"grid n={n:4d} err={err:.6e}")
    return rows


def time_convergence():
    ref_cfg = SolverConfig(nx=256, ny=256, steps=2400, dt=1.25e-4)
    ref = run_simulation(ref_cfg)["psi"]

    rows = []
    for dt, steps in [(1e-3, 300), (5e-4, 600), (2.5e-4, 1200)]:
        cfg = SolverConfig(nx=256, ny=256, steps=steps, dt=dt)
        out = run_simulation(cfg)
        err = float(jnp.sqrt(jnp.mean(jnp.abs(out["psi"] - ref) ** 2)))
        rows.append({"dt": dt, "steps": steps, "rmse_vs_ref": err})
        print(f"dt={dt:.2e} err={err:.6e}")
    return rows


def main():
    g = pd.DataFrame(grid_convergence())
    t = pd.DataFrame(time_convergence())
    g.to_csv("benchmarks/grid_convergence.csv", index=False)
    t.to_csv("benchmarks/time_convergence.csv", index=False)
    print("Saved benchmarks/grid_convergence.csv")
    print("Saved benchmarks/time_convergence.csv")


if __name__ == "__main__":
    main()
