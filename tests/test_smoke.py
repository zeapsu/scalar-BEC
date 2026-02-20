from scalar_bec.solver import SolverConfig, run_simulation


def test_smoke():
    cfg = SolverConfig(nx=64, ny=64, steps=10, dt=1e-3)
    out = run_simulation(cfg)
    assert out["psi"].shape == (64, 64)
