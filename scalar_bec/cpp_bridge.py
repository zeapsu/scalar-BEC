from __future__ import annotations
import numpy as np

try:
    import scalar_bec_cpp  # type: ignore
    HAS_CPP = True
except Exception:
    HAS_CPP = False


def apply_nonlinear_phase_cpp(psi_np: np.ndarray, V_np: np.ndarray, g: float, coeff: float):
    if not HAS_CPP:
        return False
    scalar_bec_cpp.apply_nonlinear_phase(psi_np, V_np, float(g), float(coeff))
    return True
