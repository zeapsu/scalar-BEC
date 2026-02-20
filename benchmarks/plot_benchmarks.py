from __future__ import annotations
import pandas as pd
import matplotlib.pyplot as plt


def main():
    s = pd.read_csv("benchmarks/scaling.csv")
    g = pd.read_csv("benchmarks/grid_convergence.csv")
    t = pd.read_csv("benchmarks/time_convergence.csv")

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    ax[0].plot(s["n"], s["elapsed_s"], "o-")
    ax[0].set_title("Runtime vs Grid Size")
    ax[0].set_xlabel("N")
    ax[0].set_ylabel("seconds")

    ax[1].loglog(g["n"], g["l2_error_vs_ref"], "o-")
    ax[1].set_title("Error vs Grid Size")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("L2 error")

    ax[2].loglog(t["dt"], t["rmse_vs_ref"], "o-")
    ax[2].invert_xaxis()
    ax[2].set_title("Temporal Convergence")
    ax[2].set_xlabel("dt")
    ax[2].set_ylabel("RMSE")

    fig.tight_layout()
    fig.savefig("benchmarks/benchmark_plots.png", dpi=160)
    print("Saved benchmarks/benchmark_plots.png")


if __name__ == "__main__":
    main()
