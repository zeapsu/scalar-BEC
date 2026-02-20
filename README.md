# scalar-BEC

High-performance time-splitting spectral solver for the scalar Gross–Pitaevskii equation (BEC) using **JAX** with automatic **CUDA acceleration** when available.

## Equation

We solve (dimensionless form):

\[
 i\partial_t\psi = \left[-\frac{1}{2}\nabla^2 + V(x,y) + g|\psi|^2\right]\psi
\]

with Strang split-step (time-splitting spectral):

1. Half nonlinear/potential step in real space
2. Full kinetic step in Fourier space
3. Half nonlinear/potential step in real space

## Features

- JAX + XLA JIT, runs on CPU/GPU/TPU
- FFT-based spectral kinetic propagation
- Norm and energy diagnostics
- Benchmark suite:
  - runtime vs grid size
  - error vs grid size (against high-resolution reference)
  - temporal convergence vs time step
- Optional C++ extension (`pybind11`) for fused phase multiplication kernel

## Quickstart

```bash
cd ~/projects/scalar-BEC
uv sync
```

Run a demo simulation:

```bash
uv run scalar-bec --nx 256 --ny 256 --steps 200 --dt 0.0005 --benchmark
```

Run benchmarking:

```bash
uv run python benchmarks/benchmark_scaling.py
uv run python benchmarks/benchmark_convergence.py
```

Run tests:

```bash
uv sync --group dev
uv run pytest
```

## Colab Runtime Quickstart

In a Colab-hosted runtime (including GPU runtimes), clone the repo and sync with `uv`:

```python
!git clone https://github.com/<you>/scalar-BEC.git
%cd scalar-BEC
!pip install -q uv
!uv sync
```

## CUDA notes

If JAX with CUDA is installed and a GPU is visible, code runs on GPU automatically.

Check backend:

```bash
uv run python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## Optional C++ extension

The optional `scalar_bec_cpp` extension is built as part of package installation.
To force reinstall/rebuild:

```bash
uv sync --reinstall
```

The module is `scalar_bec_cpp` and is used opportunistically in Python if available.

## Layout

- `scalar_bec/solver.py` - core split-step solver
- `scalar_bec/diagnostics.py` - norm/energy/error metrics
- `scalar_bec/cli.py` - runnable entrypoint
- `benchmarks/` - performance and convergence scripts
- `cpp/phase_kernels.cpp` - pybind11 C++ wrapper
