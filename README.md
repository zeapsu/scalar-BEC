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
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Run a demo simulation:

```bash
python -m scalar_bec.cli --nx 256 --ny 256 --steps 200 --dt 0.0005 --benchmark
```

Run benchmarking:

```bash
python benchmarks/benchmark_scaling.py
python benchmarks/benchmark_convergence.py
```

## CUDA notes

If JAX with CUDA is installed and a GPU is visible, code runs on GPU automatically.

Check backend:

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## Optional C++ extension

Build extension:

```bash
pip install -e .
```

The module is `scalar_bec_cpp` and is used opportunistically in Python if available.

## Layout

- `scalar_bec/solver.py` - core split-step solver
- `scalar_bec/diagnostics.py` - norm/energy/error metrics
- `scalar_bec/cli.py` - runnable entrypoint
- `benchmarks/` - performance and convergence scripts
- `cpp/phase_kernels.cpp` - pybind11 C++ wrapper
