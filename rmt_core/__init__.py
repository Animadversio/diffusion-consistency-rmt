"""rmt_core — random-matrix-theory toolkit for diffusion-model consistency.

Reusable machinery behind the paper "A Random Matrix Theory Perspective on the
Consistency of Diffusion Models" (ICML 2026):

- ``rmt_sc_lib``        Silverstein self-consistency solvers for kappa(z) and phi(z) (NumPy/SciPy).
- ``rmt_kappa_int_lib`` high-precision (mpmath) kappa solver + deterministic-equivalence integrals.
- ``linear_denoiser_lib`` closed-form Gaussian denoiser and Wiener-filter sampling map.
- ``random_feature_lib`` Gaussian-equivalence moments for random-feature models.
- ``DNN_sample_analysis_lib`` nearest-neighbour and experiment-parsing helpers.
"""

from . import (
    rmt_sc_lib,
    rmt_kappa_int_lib,
    linear_denoiser_lib,
    random_feature_lib,
    DNN_sample_analysis_lib,
)

from .rmt_sc_lib import (
    solve_kappa,
    solve_kappa_vectorized,
    solve_kappa_vectorized_analyCont,
    SpectrumKappa_np,
    solve_phi_newton,
    empirical_covariance_density,
)
from .rmt_kappa_int_lib import (
    SpectrumKappa_mpmath,
    solve_kappa_mpmath,
    grid1d_integrate_mat_sqrt_kappa_gauss,
    grid2d_integrate_var_kappa_gauss,
)
from .linear_denoiser_lib import (
    dnoised_X,
    wiener_gen_X,
    build_wiener_matrix,
    tsr2img,
)

__all__ = [
    "rmt_sc_lib",
    "rmt_kappa_int_lib",
    "linear_denoiser_lib",
    "random_feature_lib",
    "DNN_sample_analysis_lib",
    "solve_kappa",
    "solve_kappa_vectorized",
    "solve_kappa_vectorized_analyCont",
    "SpectrumKappa_np",
    "solve_phi_newton",
    "empirical_covariance_density",
    "SpectrumKappa_mpmath",
    "solve_kappa_mpmath",
    "grid1d_integrate_mat_sqrt_kappa_gauss",
    "grid2d_integrate_var_kappa_gauss",
    "dnoised_X",
    "wiener_gen_X",
    "build_wiener_matrix",
    "tsr2img",
]
