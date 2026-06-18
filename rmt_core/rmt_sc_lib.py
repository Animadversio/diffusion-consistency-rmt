"""Silverstein self-consistency solvers (NumPy/SciPy).

Solve the renormalized noise scale kappa(z) and the companion Stieltjes
transform phi(z) for a population spectrum {lambda_k, w_k} and aspect ratio
gamma = d/n, including analytic-continuation sweeps and a cached callable.
"""
import numpy as np
from scipy.optimize import fsolve, newton

def solve_kappa(z, eigenvalues, gamma=1.0, weights=None, initial_guess=None):
    """
    Solve for κ(z) given z and eigenvalues λ_k using the equation:
    κ(z) - z = γ * (1/p) * Σ_k [κ(z) * λ_k / (κ(z) + λ_k)]
    
    Parameters:
    - z: complex number or array
    - eigenvalues: array of eigenvalues λ_k
    - gamma: parameter γ (default 1.0)
    
    Returns:
    - κ(z): solution to the equation
    """
    if isinstance(eigenvalues, list):
        eigenvalues = np.array(eigenvalues)
    
    if weights is None:
        p = len(eigenvalues)
        weights = np.ones(p) / p
    
    def equation(kappa):
        # κ(z) - z = γ * (1/p) * Σ_k [κ(z) * λ_k / (κ(z) + λ_k)]
        sum_term = np.sum(kappa * eigenvalues * weights / (kappa + eigenvalues))
        return kappa - z - gamma * sum_term
    
    def equation_prime(kappa):
        return 1 - gamma * np.sum(eigenvalues ** 2 * weights / (kappa + eigenvalues)**2)
    
    if np.isreal(z):
        if initial_guess is None:
            initial_guess = z
        kappa_solution = fsolve(equation, initial_guess)[0]
    else:
        if initial_guess is None:
            initial_guess = z
        kappa_solution = newton(equation, initial_guess, fprime=equation_prime)
    # Solve the equation
    return kappa_solution


def solve_kappa_vectorized(z_array, eigenvalues, gamma=1.0, weights=None):
    """
    Vectorized version to solve for κ(z) for multiple z values
    """
    return np.array([solve_kappa(z, eigenvalues, gamma, weights) for z in z_array])



def solve_kappa_vectorized_analyCont(z_array, eigenvalues, gamma=1.0, weights=None, initial_guess=None):
    """
    Vectorized version to solve for κ(z) for multiple z values, esp. when z array is a continuous sequence. 
    Then we can perform "analytic continuation" by using the last kappa as the initial guess for the next z. 
    This ensures the solution is continuous along the z_array. and usually it's the correct branch of solution. 
    It's recommended to start from large real z and then go to small real z. since the solution is not stable for small z! 
    
    Note: 
    * If we want to solve for κ(z) for a continuous sequence of z, we should use this instead of solve_phi_vectorized_analyCont and then take the reciprocal. 
    The latter is not stable. 
    # IMPORTANT HEURISTIC: 
    * For this continuation, we should start from large real z and then go to small real z. since the solution is not stable for small z! 
    """
    if np.all(np.diff(z_array).real > 0):
        z_array = np.sort(z_array)[::-1]
        flip = True
    else:
        flip = False
    kappas = []
    last_kappa = initial_guess
    for z in z_array:
        kappa = solve_kappa(z, eigenvalues, gamma, weights, initial_guess=last_kappa)
        kappas.append(kappa)
        last_kappa = kappa
    if flip:
        kappas = kappas[::-1]
    return np.array(kappas)


class SpectrumKappa_np:
    def __init__(self, eigenvalues, gamma=1.0, weights=None, use_cache=True, cache_max=2000):
        self.eigenvalues = eigenvalues
        self.gamma = gamma
        self.weights = weights
        self.use_cache = use_cache
        self.cache_max = cache_max
        self.cache = {}
        
    def __call__(self, z, initial_guess=None):
        if np.isscalar(z) and z in self.cache:
            return self.cache[z]
        if initial_guess is None:
            # Use cached value with most similar key as initial guess
            if self.use_cache and self.cache:
                closest_key = min(self.cache.keys(), key=lambda k: abs(k - z))
                initial_guess = self.cache[closest_key]
            else:
                initial_guess = z
        kappa = solve_kappa(z, self.eigenvalues, self.gamma, self.weights, initial_guess=initial_guess)
        self.cache[z] = kappa
        # keep max 1000 items in cache, FIFO
        if len(self.cache) >= self.cache_max:
            self.cache.pop(next(iter(self.cache)))
        return kappa
    
    def solve_path_analyCont(self, z_list, initial_guess=None):
        kappas = solve_kappa_vectorized_analyCont(z_list, self.eigenvalues, self.gamma, self.weights, initial_guess=initial_guess)
        for z, kappa in zip(z_list, kappas):
            self.cache[z] = kappa
        return kappas




def solve_phi_newton(z, eigenvalues, gamma, weights=None, phi0=None, maxiter=500, tol=1e-12, solver='newton_custom'):
    """
    Solve 1-gamma + z*phi + gamma*sum_i weights[i]/(phi*taus[i]+1) = 0
    for complex phi, using Newton's method.
    taus: array of eigenvalues (tau_i >= 0)
    weights: same shape, weights summing to 1 (dF_Sigma)
    z: complex
    gamma: scalar in (0,1)
    phi0: initial complex guess (default: 1j/z)
    """
    if phi0 is None:
        if not np.isreal(z):
            phi0 = 1j * np.abs(1.0/z)
        else:
            phi0 = 1.0/z

    if isinstance(eigenvalues, list):
        eigenvalues = np.array(eigenvalues)
    
    if weights is None:
        weights = np.ones(len(eigenvalues)) / len(eigenvalues)
    
    def H(phi):
        return (1 - gamma) + z * phi + gamma * np.sum(weights / (phi * eigenvalues + 1))

    def Hprime(phi):
        # derivative w.r.t. phi
        return z - gamma * np.sum(weights * eigenvalues / (phi * eigenvalues + 1)**2)

    # return newton(H, phi0, fprime=Hprime, tol=1e-12, maxiter=500)
    if solver == 'newton_custom':
        phi = phi0
        for i in range(maxiter):
            f = H(phi)
            if abs(f) < tol:
                break

            step = f / Hprime(phi)
            phi_next = phi - step

            # 2) enforce Im(phi)>=0 by backtracking + reflection
            if phi_next.imag < 0:
                alpha = 1.0
                while alpha > 1e-6 and (phi - alpha*step).imag < 0:
                    alpha *= 0.5
                phi_next = phi - alpha*step
                if phi_next.imag < 0:
                    phi_next = phi_next.real + 1j*abs(phi_next.imag)

            phi = phi_next
        return phi
    elif solver == 'newton':
        return newton(H, phi0, fprime=Hprime, tol=tol, maxiter=maxiter)
    elif solver == 'fsolve':
        return fsolve(H, phi0, xtol=tol, maxfev=maxiter)
    else:
        raise ValueError(f"Invalid solver: {solver}")
    

def solve_phi_vectorized(z_array, eigenvalues, gamma=1.0, weights=None, **kwargs):
    """
    Vectorized version to solve for φ(z) for multiple z values
    """
    return np.array([solve_phi_newton(z, eigenvalues, gamma, weights, **kwargs) for z in z_array])


def solve_phi_vectorized_analyCont(z_array, eigenvalues, gamma=1.0, weights=None, initial_guess=None, **kwargs):
    """
    Vectorized version to solve for κ(z) for multiple z values
    """
    # assert np.all(np.diff(z_array).real <= 0), 'z_array should be sorted in descending order, or the solver could be unstable for small z'
    if np.all(np.diff(z_array).real > 0):
        z_array = np.sort(z_array)[::-1]
        flip = True
    else:
        flip = False
    last_phi = initial_guess
    phis = []
    for z in z_array:
        phi = solve_phi_newton(z, eigenvalues, gamma, weights, phi0=last_phi, **kwargs)
        phis.append(phi)
        last_phi = phi
    if flip:
        phis = phis[::-1]
    return np.array(phis)


def empirical_covariance_density(xvec, eigenvalues, gamma, weights=None, eps=1e-6, **kwargs):
    """
    Compute the empirical density of the eigenvalues
    """
    zvec = xvec+eps*1j
    phis = solve_phi_vectorized_analyCont(zvec, eigenvalues, gamma, weights, **kwargs)
    m_sol = 1 / zvec * (1 / gamma - 1) + 1 / gamma*phis
    return m_sol.imag / np.pi
