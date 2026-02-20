from __future__ import annotations
import time
import pandas as pd
from scalar_bec.solver import SolverConfig, run_simulation


def run_once(nx, ny, steps=300, dt=5e-4):
    cfg = SolverConfig(nx=nx, ny=ny, steps=steps, dt=dt)
    t0 = time.perf_counter()
    out = run_simulation(cfg)
    out["psi"].block_until_ready()
    t1 = time.perf_counter()
    return t1 - t0


def main():
    grids = [64, 128, 192, 256, 384, 512]
    rows = []
    for n in grids:
        t = run_once(n, n)
        rows.append({"n": n, "elapsed_s": t, "points": n*n, "points_per_s": (n*n)/t})
        print(f"n={n:4d} elapsed_s={t:.5f} points_per_s={(n*n)/t:.3e}")

    df = pd.DataFrame(rows)
    df.to_csv("benchmarks/scaling.csv", index=False)
    print("Saved benchmarks/scaling.csv")


if __name__ == "__main__":
    main()
