"""High-precision kappa solvers and deterministic-equivalence integrals.

An mpmath-backed callable for kappa(z) with continuation/caching, plus the
Gauss-Legendre quadratures (with a tan-transform to [0, inf)) that evaluate the
fractional-power deterministic-equivalence integrals for the sampling map's
expectation and variance.
"""
import numpy as np
import mpmath as mp

class SpectrumKappa_mpmath:
    """
    Callable kappa(z) backed by a spectrum {lambda_k, w_k}, using mp.findroot.
    Extras:
      - warmup() to solve at a large z (or user-chosen z) to prime the cache
      - solve_path(z_list) to solve sequentially, always seeding from the nearest
        previously-solved z (continuation + nearest-neighbor)
    """
    def __init__(self, eigenvalues, gamma=1.0, weights=None,
                 dps=80, maxsteps=80, tol=None, catalog_max=2048):
        self.gamma = mp.mpf(gamma)
        self.dps = int(dps)
        self.maxsteps = int(maxsteps)
        self.tol = tol  # if None, defaults to mp.eps at call time
        self.catalog_max = int(catalog_max)

        # prep eigs/weights
        self.eigs = [mp.mpf(l) for l in eigenvalues]
        p = len(self.eigs)
        if weights is None:
            self.wts = [mp.mpf(1)/p]*p
        else:
            self.wts = [mp.mpf(w) for w in weights]
            s = mp.fsum(self.wts)
            if s != 1:
                self.wts = [w/s for w in self.wts]

        # moment for large-z asymptotic
        self.mu1 = mp.fsum(w*l for w, l in zip(self.wts, self.eigs))

        # continuation hints
        self._last_z = None
        self._last_k = None

        # global catalog of solutions: list of (z, kappa)
        self._catalog = []

    # ---------- core residuals ----------
    def _F(self, kappa, z):
        return kappa - z - self.gamma * mp.fsum(
            w*(kappa*l)/(kappa + l) for w, l in zip(self.wts, self.eigs)
        )

    def _Fprime(self, kappa):
        return 1 - self.gamma * mp.fsum(
            w*(l*l)/(kappa + l)**2 for w, l in zip(self.wts, self.eigs)
        )

    # ---------- initial guesses ----------
    def _good_initial(self, z):
        g0 = z + self.gamma * self.mu1
        eps = mp.mpf('1e-6')
        for l in self.eigs:
            if abs(g0 + l) < eps:
                g0 += (1 + 1j) * eps
        return g0

    def _nearest_from_catalog(self, z):
        """Return (z0, k0) for the catalog entry with smallest |z - z0|, or (None, None)."""
        if not self._catalog:
            return None, None
        z = mp.mpc(z) if (hasattr(z, "imag") and z.imag != 0) else mp.mpf(z)
        j = min(range(len(self._catalog)), key=lambda i: abs(z - self._catalog[i][0]))
        return self._catalog[j]

    def _remember(self, z, kappa):
        """Store (z,kappa) in the catalog and keep it bounded."""
        self._catalog.append((z, kappa))
        if len(self._catalog) > self.catalog_max:
            self._catalog.pop(0)  # FIFO

    # ---------- main callable ----------
    def __call__(self, z, dps_override=None, maxsteps_override=None, tol=None):
        """Return kappa(z). Accepts real or complex z."""
        old_dps = mp.mp.dps
        mp.mp.dps = int(dps_override or self.dps)
        try:
            z = mp.mpc(z) if (hasattr(z, "imag") and z.imag != 0) else mp.mpf(z)
            tol = self.tol if tol is None else tol

            # pick initial guess: nearest catalog → last_z (if close) → asymptotic
            z0, k0 = self._nearest_from_catalog(z)
            if z0 is not None:
                x0 = k0
            elif self._last_z is not None and abs(z - self._last_z) <= 1:
                x0 = self._last_k
            else:
                x0 = self._good_initial(z)

            # Newton with derivative; fallback to secant
            try:
                kappa = mp.findroot(lambda k: self._F(k, z),
                                    x0,
                                    df=lambda k: self._Fprime(k),
                                    tol=(tol or mp.eps),
                                    maxsteps=int(maxsteps_override or self.maxsteps))
            except Exception:
                x1 = x0*(1 + mp.mpf('1e-6')) if x0 else mp.mpf('1e-6')
                kappa = mp.findroot(lambda k: self._F(k, z),
                                    x0, x1,
                                    tol=(tol or mp.eps),
                                    maxsteps=int(maxsteps_override or self.maxsteps))

            # update hints + catalog
            self._last_z, self._last_k = z, kappa
            self._remember(z, kappa)
            return kappa
        finally:
            mp.mp.dps = old_dps

    # ---------- utilities ----------
    def warmup(self, warmup_z=None, scale=1e6, **call_kwargs):
        """
        Solve kappa at a large z to 'prime' the cache (or at user-provided warmup_z).
        Returns (z_warm, kappa(z_warm)).
        """
        if warmup_z is None:
            # heuristic: a very large real z
            warmup_z = mp.mpf(scale)
        kw = dict(call_kwargs)
        k = self(warmup_z, **kw)
        return warmup_z, k

    def solve_path(self, z_list, reverse_if_increasing_real=True, warmup_z=None,
                   use_warmup_if_empty=True, **call_kwargs):
        """
        Solve kappa for many z values:
          - If the list's real parts are strictly increasing and reverse_if_increasing_real=True,
            solve in reverse (large → small) to encourage a stable branch.
          - Each new z seeds from the closest previously solved z in the catalog.

        Returns a list of kappas in the SAME ORDER as input z_list.
        """
        # optional warmup
        if use_warmup_if_empty and not self._catalog and warmup_z is None:
            self.warmup()  # default large-z warmup
        elif warmup_z is not None:
            self.warmup(warmup_z)

        # decide sweep direction (for stability)
        z_seq = list(z_list)
        def strictly_increasing_re(xs):
            reals = [mp.re(mp.mpc(z)) for z in xs]
            return all(reals[i+1] > reals[i] for i in range(len(reals)-1))
        flipped = False
        if reverse_if_increasing_real and len(z_seq) >= 2 and strictly_increasing_re(z_seq):
            z_seq = z_seq[::-1]
            flipped = True

        # solve sequentially (catalog will be filled and reused)
        kappas_seq = [self(z, **call_kwargs) for z in z_seq]

        # restore original order
        if flipped:
            kappas_seq.reverse()
        return kappas_seq


def solve_kappa_mpmath(z, eigenvalues, gamma=1.0, weights=None, initial_guess=None,
                dps=80, maxsteps=50):
    old_dps = mp.mp.dps
    mp.mp.dps = dps
    try:
        # cast
        z = mp.mpc(z) if (hasattr(z, "imag") and z.imag != 0) else mp.mpf(z)
        gamma = mp.mpf(gamma)

        # prep eigs/weights
        if hasattr(eigenvalues, "__iter__"):
            eigs = [mp.mpf(l) for l in eigenvalues]
        else:
            eigs = [mp.mpf(eigenvalues)]
        p = len(eigs)
        if weights is None:
            wts = [mp.mpf(1)/p]*p
        else:
            wts = [mp.mpf(w) for w in weights]
            s = mp.fsum(wts)
            if s != 1:
                wts = [w/s for w in wts]

        def F(kappa):
            return kappa - z - gamma * mp.fsum(
                w * (kappa*lam)/(kappa + lam) for lam, w in zip(eigs, wts)
            )

        def Fprime(kappa):
            return 1 - gamma * mp.fsum(
                w * (lam**2)/(kappa + lam)**2 for lam, w in zip(eigs, wts)
            )

        # default initial guess
        if initial_guess is None:
            g0 = z
            eps = mp.mpf('1e-6')
            for lam in eigs:
                if abs(g0 + lam) < eps:
                    g0 += (1 + 1j) * eps
            initial_guess = g0

        #  pass derivative via keyword df=...
        try:
            kappa = mp.findroot(F, initial_guess, df=Fprime, tol=mp.eps, maxsteps=maxsteps)
        except Exception:
            # fallback: secant with a tiny perturbation
            kappa = mp.findroot(F, initial_guess, initial_guess*(1+mp.mpf('1e-6')),
                                tol=mp.eps, maxsteps=maxsteps)
        return kappa
    finally:
        mp.mp.dps = old_dps


def solve_kappa_vectorized_mpmath(z_array, eigenvalues, gamma=1.0, weights=None,
                           dps=80, maxsteps=50, initial_guess=None):
    """
    Apply solve_kappa to each z in z_array independently.
    """
    out = []
    last_init = initial_guess
    for z in z_array:
        out.append(solve_kappa_mpmath(z, eigenvalues, gamma, weights,
                               initial_guess=last_init if last_init is not None else z,
                               dps=dps, maxsteps=maxsteps))
    return out


def solve_kappa_vectorized_analyCont_mpmath(z_array, eigenvalues, gamma=1.0, weights=None,
                                     dps=80, maxsteps=50, initial_guess=None,
                                     reverse_if_increasing_real=True):
    """
    Analytic continuation along a path of z values:
    - Uses the previous solution as the initial guess for the next z.
    - Optionally reverses the path if Re(z) is strictly increasing (often
      better to sweep from large to small Re(z)).

    Returns a list of kappa values matching the original z order.
    """
    # Convert to list so we can possibly reverse and then restore
    z_list = list(z_array)

    def strictly_increasing_re(xs):
        # Check strictly increasing real parts
        reals = [mp.re(mp.mpc(z)) for z in xs]
        return all(reals[i+1] > reals[i] for i in range(len(reals)-1))

    flipped = False
    if reverse_if_increasing_real and len(z_list) >= 2 and strictly_increasing_re(z_list):
        z_list = z_list[::-1]
        flipped = True

    kappas = []
    last_kappa = initial_guess
    for i, z in enumerate(z_list):
        kappa = solve_kappa_mpmath(z, eigenvalues, gamma, weights,
                            initial_guess=(last_kappa if last_kappa is not None else z),
                            dps=dps, maxsteps=maxsteps)
        kappas.append(kappa)
        last_kappa = kappa

    if flipped:
        kappas.reverse()
    return kappas


########################
### Old versions integraion, performing integral on infinite domain, which has large truncation error. No longer used.
########################

def _trapz_weights(x):
    """Trapezoid weights for a 1D, strictly increasing grid x (not necessarily uniform)."""
    w = np.empty_like(x, dtype=float)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0]     = 0.5 * (x[1] - x[0])
    w[-1]    = 0.5 * (x[-1] - x[-2])
    return w


def _grid2d_integrate_double_kappa(
    kappa,                     # callable: kappa(s) for s >= 0
    Sigma,                     # scalar Σ
    sigma,                     # lower limit for u
    u_max, v_max,              # finite truncation limits for u, v
    n_u=400, n_v=400,          # grid sizes
    use_log_u=True, use_log_v=True,
    uv_min=1e-8,      # smallest positive v for log-part of v-grid (we also include v=0)
    diag_tol_rel=1e-8          # relative tolerance for |u^2 - v^2| to use diagonal limit
):
    """
    Numerically approximates:
      (4/pi^2) * ∫_{u=sigma..∞} ∫_{v=0..∞}
         [(kappa(u^2) - kappa(v^2)) / (u^2 - v^2)]
         * Σ (Σ + kappa(u^2))^{-1} (Σ + kappa(v^2))^{-1}  dv du
    using finite truncation to [sigma, u_max] × [0, v_max].
    """
    if Sigma == 0:
        return 0.0

    # --- 1) Build u, v grids (log grids default; v includes an exact 0 endpoint) ---
    if use_log_u:
        u = np.geomspace(uv_min, u_max, n_u)
    else:
        u = np.linspace(0.0, u_max, n_u)

    if use_log_v:
        # include v=0 explicitly + log-spaced positive tail
        v_pos = np.geomspace(uv_min, v_max, max(n_v-1, 1))
        v = np.concatenate(([0.0], v_pos))
    else:
        v = np.linspace(0.0, v_max, n_v)

    wu = _trapz_weights(u)    # du weights
    wv = _trapz_weights(v)    # dv weights

    # --- 2) Prepare s = u^2 and t = v^2, evaluate kappa there ---
    s = u**2 + sigma**2
    t = v**2

    # vectorized evaluation (works even if kappa is scalar-only via np.vectorize)
    kappa_vec = np.vectorize(lambda x: float(kappa(float(x))), otypes=[np.float64])
    Ku = kappa_vec(s)    # shape (n_u,)
    Kv = kappa_vec(t)    # shape (n_v,)

    # --- 3) Build the kernel safely near the diagonal u^2 ≈ v^2 ---
    # Raw quotient
    denom = s[:, None] - t[None, :]
    num   = Ku[:, None] - Kv[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        frac = num / denom

    # Diagonal mask where we should use the limit kappa'(s)
    # relative tolerance scaled by max(s, t) to be scale-aware
    scale = np.maximum(s[:, None], t[None, :])
    diag_mask = np.abs(denom) <= (diag_tol_rel * np.maximum(scale, 1.0))

    # Precompute kappa'(s) on the u-grid via central differences in s
    # (more stable than per-pair derivatives)
    dKu_ds = np.gradient(Ku, s, edge_order=2)  # shape (n_u,)

    # Fill near-diagonal entries with the limit value using dKu_ds at the u-location
    # (equally valid to use v-location; this choice is consistent and fast)
    frac[diag_mask] = dKu_ds[np.nonzero(diag_mask)[0]]

    # --- 4) Resolvent factors and full integrand matrix ---
    # Σ / ((Σ + kappa(u^2)) (Σ + kappa(v^2)))
    R = Sigma / ((Sigma + Ku)[:, None] * (Sigma + Kv)[None, :])

    integrand = frac * R  # shape (n_u, n_v)

    # --- 5) 2D integration via separable trapezoid weights (nonuniform OK) ---
    I_rect = np.sum(integrand * (wu[:, None] * wv[None, :]))  # ≈ ∫∫ f(u,v) dv du

    # --- 6) Overall prefactor ---
    val = (4.0 / (np.pi**2)) * I_rect
    return val


########################
### New versions integraion, performing integral on finite domain, using gauss-legendre quadrature, tanh tansform, which has no truncation error.
########################


# Final version using Gauss-Legendre quadrature, tanh transform, no truncation error.
# 1D integration for matrix square root deterministic equivalent

def gauss_legendre_on_0_pi2(n):
    # nodes/weights for theta in [0, pi/2]
    x, w = np.polynomial.legendre.leggauss(n)   # on [-1, 1]
    th = 0.25*np.pi*(x + 1.0)                   # map to [0, pi/2]
    wt = 0.25*np.pi * w
    return th, wt


def tan_to_0_inf_transform(th):
    u = np.tan(th)
    jac = 1 / np.cos(th)**2
    return u, jac


def grid1d_integrate_mat_sqrt_kappa_gauss(kappa, Sigma_vec, eigvec_proj=None, n_nodes=100):
    """
    Compute the matrix square root using the integral formula:
    Σ^{1/2} = (2/π) ∫₀^∞ Σ(Σ + κ(u²)I)^{-1} du
    
    where κ is a function of u, Sigma_vec is a vector of eigenvalues,
    and n_nodes is the number of nodes in the integration.
    
    kappa: is a function support calling and vectorization. 
    """
    th, wt = gauss_legendre_on_0_pi2(n_nodes)
    u, jac = tan_to_0_inf_transform(th)

    def _vectorize_kappa(kappa):
        # robust vectorizer (keeps scalar call signature)
        return np.vectorize(lambda x: float(kappa(float(x))), otypes=[np.float64])

    kappa_u = _vectorize_kappa(kappa)(u**2)
    
    # Compute Σ(Σ + κ(u²)I)^{-1} for each u
    integrand = Sigma_vec[:, None] / (Sigma_vec[:, None] + kappa_u[None, :])
    if eigvec_proj is not None:
        integrand = integrand * eigvec_proj[:, None]**2
    
    # Integrate over u with proper Jacobian and weights
    weighted_integrand = integrand * wt[None, :] * jac[None, :]
    integration_result = weighted_integrand.sum(axis=1)
    
    # Apply the coefficient 2/π
    integration_result = integration_result * 2 / np.pi
    
    return integration_result


def grid2d_integrate_var_kappa_gauss(kappa, Sigma_vec, lambda_k, n_nodes=100, n_sample=100):
    """
    Integration of 
    $$
    4/\pi^2\int_{0}^{\infty}\int_{0}^{\infty}\Big\{\frac{\kappa\kappa^{\prime}\mathrm{Tr}[\Sigma(\Sigma+\kappa I)^{-1}(\Sigma+\kappa^{\prime}I)^{-1}]}{n-\mathrm{df}_{2}(\kappa,\kappa^{\prime})}\big[\mathbf{v}^{\top}\Sigma(\Sigma+\kappa I)^{-1}(\Sigma+\kappa^{\prime}I)^{-1}\mathbf{v}\big]\!\Big\}\!dudv
    $$
    where $\kappa$ is a function of $u$ and $v$, $\Sigma$ is a vector of eigenvalues, $\lambda_k$ is a scalar, $n$ is the number of samples, and $n_nodes$ is the number of nodes in the integration.
    """
    lambda_k_vec = np.atleast_1d(np.asarray(lambda_k, dtype=float))
    th, wt = gauss_legendre_on_0_pi2(n_nodes)
    u, jac = tan_to_0_inf_transform(th)

    def _vectorize_kappa(kappa):
        # robust vectorizer (keeps scalar call signature)
        return np.vectorize(lambda x: float(kappa(float(x))), otypes=[np.float64])

    kappa_u = _vectorize_kappa(kappa)( u**2 ) 
    kappa_v = np.copy(kappa_u)
    weight_grid = wt[:,None] * wt[None,:]
    jac_grid = jac[:,None] * jac[None,:]
    # 
    df2_val = ( Sigma_vec[:,None,None] ** 2 / (Sigma_vec[:,None,None] + kappa_u[None,:,None]) / (Sigma_vec[:,None,None] + kappa_v[None,None,:]) ).sum(axis=0)
    numerator_trace = ( Sigma_vec[:,None,None] / (Sigma_vec[:,None,None] + kappa_u[None,:,None]) / (Sigma_vec[:,None,None] + kappa_v[None,None,:]) ).sum(axis=0)
    scalar_grid = kappa_u[:,None] * kappa_v[None,:] * numerator_trace / (n_sample - df2_val) #* lambda_k / (lambda_k + kappa_u[None,:,None]) / (lambda_k + kappa_v[None,None,:])
    eigen_bilinear_form = lambda_k_vec[:,None,None] / (lambda_k_vec[:,None,None] + kappa_u[None,:,None]) / (lambda_k_vec[:,None,None] + kappa_v[None,None,:])
    final_grid = eigen_bilinear_form * (scalar_grid * weight_grid * jac_grid)[None,:,:]
    integration_result = final_grid.sum(axis=(1,2))
    integration_result = integration_result * 4 / np.pi**2
    return integration_result
