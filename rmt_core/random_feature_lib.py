"""
Random Feature Gaussian Equivalence Library

This module implements the random feature approximation formula:
φ(Fx) ≈ μ₀1 + μ₁Fx + ξ, where ξ ~ N(0, μ*I)

The library provides functions for:
- Computing moments of activation functions
- Generating original and approximated random features
- Validating statistical equivalence
- Various activation function implementations
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Callable, Optional
import warnings
from scipy import integrate


def compute_moments(activation_func: Callable[[torch.Tensor], torch.Tensor], 
                   num_samples: int = 1000000,
                   device: str = "cpu") -> Tuple[float, float, float]:
    """
    Compute the moments μ₀, μ₁, μ* for a given activation function φ.
    
    Args:
        activation_func: Function that takes a tensor and returns φ(x)
        num_samples: Number of samples for Monte Carlo estimation
        device: Device to run computations on
    
    Returns:
        tuple: (μ₀, μ₁, μ*) where:
            - μ₀ = E[φ(x)] for x ~ N(0,1)
            - μ₁ = E[x*φ(x)] for x ~ N(0,1)  
            - μ* = E[φ(x)²] - μ₀² - μ₁²
    """
    # Sample from standard normal distribution
    x = torch.randn(num_samples, device=device)
    
    # Compute φ(x)
    phi_x = activation_func(x)
    
    # Compute moments
    mu_0 = torch.mean(phi_x).item()
    mu_1 = torch.mean(x * phi_x).item()
    phi_x_squared = torch.mean(phi_x**2).item()
    mu_star = phi_x_squared - mu_0**2 - mu_1**2
    
    # Ensure μ* is non-negative (should be by construction, but numerical errors)
    if mu_star < 0:
        warnings.warn(f"μ* = {mu_star:.6f} < 0, setting to 0. This may indicate numerical issues.")
        mu_star = 0.0
    
    return mu_0, mu_1, mu_star


def compute_moments_integration(activation_func: Callable[[torch.Tensor], torch.Tensor],
                               bounds: Tuple[float, float] = (-8.0, 8.0),
                               epsabs: float = 1e-8,
                               epsrel: float = 1e-8) -> Tuple[float, float, float]:
    """
    Compute the moments μ₀, μ₁, μ* for a given activation function φ using numerical integration.
    
    Args:
        activation_func: Function that takes a tensor and returns φ(x)
        bounds: Integration bounds (default: (-8, 8) covers ~99.999% of standard normal)
        epsabs: Absolute error tolerance for integration
        epsrel: Relative error tolerance for integration
    
    Returns:
        tuple: (μ₀, μ₁, μ*) where:
            - μ₀ = ∫ φ(x) * (1/√(2π)) * exp(-x²/2) dx
            - μ₁ = ∫ x*φ(x) * (1/√(2π)) * exp(-x²/2) dx  
            - μ* = ∫ φ(x)² * (1/√(2π)) * exp(-x²/2) dx - μ₀² - μ₁²
    """
    # Standard normal PDF: (1/√(2π)) * exp(-x²/2)
    sqrt_2pi = np.sqrt(2 * np.pi)
    
    def standard_normal_pdf(x):
        return np.exp(-0.5 * x**2) / sqrt_2pi
    
    # Create numpy-compatible activation function
    def phi_numpy(x):
        # Convert to tensor, apply activation, convert back to numpy
        if np.isscalar(x):
            x_tensor = torch.tensor(float(x))
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32)
        
        with torch.no_grad():
            result = activation_func(x_tensor)
        
        if np.isscalar(x):
            return result.item()
        else:
            return result.numpy()
    
    # Define integrands
    def mu_0_integrand(x):
        return phi_numpy(x) * standard_normal_pdf(x)
    
    def mu_1_integrand(x):
        return x * phi_numpy(x) * standard_normal_pdf(x)
    
    def phi_squared_integrand(x):
        return phi_numpy(x)**2 * standard_normal_pdf(x)
    
    # Compute integrals
    try:
        mu_0, _ = integrate.quad(mu_0_integrand, bounds[0], bounds[1], 
                                epsabs=epsabs, epsrel=epsrel)
        mu_1, _ = integrate.quad(mu_1_integrand, bounds[0], bounds[1], 
                                epsabs=epsabs, epsrel=epsrel)
        phi_x_squared, _ = integrate.quad(phi_squared_integrand, bounds[0], bounds[1], 
                                         epsabs=epsabs, epsrel=epsrel)
        
        mu_star = phi_x_squared - mu_0**2 - mu_1**2
        
        # Ensure μ* is non-negative
        if mu_star < 0:
            warnings.warn(f"μ* = {mu_star:.6f} < 0, setting to 0. This may indicate numerical issues.")
            mu_star = 0.0
            
    except integrate.IntegrationWarning as e:
        warnings.warn(f"Integration warning: {e}")
        raise
    except Exception as e:
        raise RuntimeError(f"Integration failed: {e}")
    
    return mu_0, mu_1, mu_star


def validate_moment_computation_methods(activation_func: Callable[[torch.Tensor], torch.Tensor],
                                       activation_name: str = "unknown",
                                       num_samples: int = 1000000,
                                       bounds: Tuple[float, float] = (-8.0, 8.0),
                                       device: str = "cpu",
                                       verbose: bool = True) -> Dict[str, float]:
    """
    Compare Monte Carlo and integration-based moment computation methods.
    
    Args:
        activation_func: Activation function to test
        activation_name: Name for display purposes
        num_samples: Number of samples for Monte Carlo method
        bounds: Integration bounds for numerical integration
        device: Device for Monte Carlo computation
        verbose: Whether to print detailed comparison
        
    Returns:
        Dictionary with error metrics between the two methods
    """
    if verbose:
        print(f"=== Moment Computation Method Validation: {activation_name.upper()} ===")
    
    # Compute moments using Monte Carlo method
    mc_start_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    mc_end_time = torch.cuda.Event(enable_timing=True) if device == "cuda" else None
    
    if device == "cuda":
        mc_start_time.record()
    import time
    start_time = time.time()
    
    mu_0_mc, mu_1_mc, mu_star_mc = compute_moments(activation_func, num_samples, device)
    
    if device == "cuda":
        mc_end_time.record()
        torch.cuda.synchronize()
        mc_time = mc_start_time.elapsed_time(mc_end_time) / 1000.0  # Convert to seconds
    else:
        mc_time = time.time() - start_time
    
    # Compute moments using integration method
    start_time = time.time()
    mu_0_int, mu_1_int, mu_star_int = compute_moments_integration(activation_func, bounds)
    int_time = time.time() - start_time
    
    # Compute relative errors
    def relative_error(a, b):
        if abs(a) < 1e-10:
            return abs(a - b)
        return abs(a - b) / abs(a)
    
    mu_0_error = relative_error(mu_0_int, mu_0_mc)
    mu_1_error = relative_error(mu_1_int, mu_1_mc)
    mu_star_error = relative_error(mu_star_int, mu_star_mc)
    
    if verbose:
        print(f"\nMonte Carlo Results (n={num_samples:,}):")
        print(f"  μ₀ = {mu_0_mc:.8f}")
        print(f"  μ₁ = {mu_1_mc:.8f}")
        print(f"  μ* = {mu_star_mc:.8f}")
        print(f"  Computation time: {mc_time:.4f}s")
        
        print(f"\nNumerical Integration Results (bounds={bounds}):")
        print(f"  μ₀ = {mu_0_int:.8f}")
        print(f"  μ₁ = {mu_1_int:.8f}")
        print(f"  μ* = {mu_star_int:.8f}")
        print(f"  Computation time: {int_time:.4f}s")
        
        print(f"\nRelative Errors:")
        print(f"  μ₀ error: {mu_0_error:.2e} ({mu_0_error*100:.4f}%)")
        print(f"  μ₁ error: {mu_1_error:.2e} ({mu_1_error*100:.4f}%)")
        print(f"  μ* error: {mu_star_error:.2e} ({mu_star_error*100:.4f}%)")
        
        print(f"\nSpeed comparison:")
        if mc_time > 0:
            speedup = mc_time / int_time if int_time > 0 else float('inf')
            if speedup > 1:
                print(f"  Integration is {speedup:.1f}x faster than Monte Carlo")
            else:
                print(f"  Monte Carlo is {1/speedup:.1f}x faster than Integration")
        
        # Overall assessment
        max_error = max(mu_0_error, mu_1_error, mu_star_error)
        if max_error < 1e-4:
            print(f"\n✓ Excellent agreement (max error: {max_error:.2e})")
        elif max_error < 1e-3:
            print(f"\n✓ Good agreement (max error: {max_error:.2e})")
        elif max_error < 1e-2:
            print(f"\n⚠ Fair agreement (max error: {max_error:.2e})")
        else:
            print(f"\n✗ Poor agreement (max error: {max_error:.2e})")
    
    return {
        'mu_0_mc': mu_0_mc,
        'mu_1_mc': mu_1_mc,
        'mu_star_mc': mu_star_mc,
        'mu_0_int': mu_0_int,
        'mu_1_int': mu_1_int,
        'mu_star_int': mu_star_int,
        'mu_0_error': mu_0_error,
        'mu_1_error': mu_1_error,
        'mu_star_error': mu_star_error,
        'mc_time': mc_time,
        'int_time': int_time,
        'max_error': max(mu_0_error, mu_1_error, mu_star_error)
    }


def batch_validate_moment_methods(activation_names: Optional[list] = None,
                                  num_samples: int = 1000000,
                                  bounds: Tuple[float, float] = (-8.0, 8.0),
                                  device: str = "cpu") -> Dict:
    """
    Batch validation of Monte Carlo vs Integration methods for multiple activation functions.
    
    Args:
        activation_names: List of activation function names to test
        num_samples: Number of samples for Monte Carlo method
        bounds: Integration bounds for numerical integration
        device: Device for Monte Carlo computation
        
    Returns:
        Dictionary with validation results for each activation function
    """
    if activation_names is None:
        activation_names = list(ACTIVATION_FUNCTIONS.keys())
    
    results = {}
    
    print("=" * 80)
    print("BATCH VALIDATION: MONTE CARLO vs NUMERICAL INTEGRATION")
    print("=" * 80)
    
    for name in activation_names:
        print(f"\n{'-' * 50}")
        activation_func = ACTIVATION_FUNCTIONS[name]
        result = validate_moment_computation_methods(
            activation_func, name, num_samples, bounds, device, verbose=True
        )
        results[name] = result
    
    # Summary table
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Activation':<12} | {'Max Error':<10} {'MC Time':<8} {'Int Time':<8} {'Speedup':<8} | {'Assessment'}")
    print("-" * 80)
    
    for name, data in results.items():
        max_err = data['max_error']
        mc_time = data['mc_time']
        int_time = data['int_time']
        speedup = mc_time / int_time if int_time > 0 else float('inf')
        
        if max_err < 1e-4:
            assessment = "Excellent"
        elif max_err < 1e-3:
            assessment = "Good"
        elif max_err < 1e-2:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        print(f"{name:<12} | {max_err:<10.2e} {mc_time:<8.3f} {int_time:<8.3f} {speedup:<8.1f} | {assessment}")
    
    return results


def random_feature_original(X: torch.Tensor, 
                          F: torch.Tensor, 
                          activation_func: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
    """
    Original random feature model: φ(FX)
    
    Args:
        X: Input data (n_samples, input_dim)
        F: Random feature matrix (feature_dim, input_dim)
        activation_func: Activation function φ
    
    Returns:
        φ(FX): Random features (n_samples, feature_dim)
    """
    FX = torch.matmul(X, F.T)  # (n_samples, feature_dim)
    return activation_func(FX)


def random_feature_approximation(X: torch.Tensor,
                                F: torch.Tensor, 
                                mu_0: float, 
                                mu_1: float, 
                                mu_star: float) -> torch.Tensor:
    """
    Approximated random feature model: μ₀1 + μ₁FX + ξ
    
    Args:
        X: Input data (n_samples, input_dim) 
        F: Random feature matrix (feature_dim, input_dim)
        mu_0, mu_1, mu_star: Computed moments
    
    Returns:
        Approximated features (n_samples, feature_dim)
    """
    # Linear part: μ₁FX
    FX = torch.matmul(X, F.T)  # (n_samples, feature_dim)
    linear_part = mu_1 * FX
    
    # Constant part: μ₀1
    constant_part = mu_0 * torch.ones_like(linear_part)
    
    # Noise part: ξ ~ N(0, μ*I)
    if mu_star > 0:
        noise = torch.sqrt(torch.tensor(mu_star, device=X.device)) * torch.randn_like(linear_part)
    else:
        noise = torch.zeros_like(linear_part)
    
    return constant_part + linear_part + noise


def validate_equivalence(features_orig: torch.Tensor, 
                        features_approx: torch.Tensor,
                        verbose: bool = True) -> Dict[str, float]:
    """
    Validate statistical equivalence between original and approximated features.
    
    Args:
        features_orig: Original random features
        features_approx: Approximated random features
        verbose: Whether to print detailed validation results
    
    Returns:
        Dictionary with error metrics: mean_error, var_error, cov_error, dist_error
    """
    if verbose:
        print("=== Statistical Equivalence Validation ===")
    
    # Test 1: Mean comparison
    mean_orig = torch.mean(features_orig)
    mean_approx = torch.mean(features_approx)
    mean_error = abs(mean_orig - mean_approx).item()
    
    if verbose:
        print(f"1. Mean comparison:")
        print(f"   Original: {mean_orig.item():.6f}")
        print(f"   Approximated: {mean_approx.item():.6f}")
        rel_error = abs(mean_orig - mean_approx) / abs(mean_orig) * 100 if abs(mean_orig) > 1e-8 else 0
        print(f"   Relative error: {rel_error:.2f}%")
    
    # Test 2: Variance comparison
    var_orig = torch.var(features_orig)
    var_approx = torch.var(features_approx)
    var_error = abs(var_orig - var_approx).item()
    
    if verbose:
        print(f"\n2. Variance comparison:")
        print(f"   Original: {var_orig.item():.6f}")
        print(f"   Approximated: {var_approx.item():.6f}")
        rel_error = abs(var_orig - var_approx) / abs(var_orig) * 100 if abs(var_orig) > 1e-8 else 0
        print(f"   Relative error: {rel_error:.2f}%")
    
    # Test 3: Covariance matrix comparison (sample a subset for efficiency)
    subset_size = min(50, features_orig.shape[1])
    indices = torch.randperm(features_orig.shape[1])[:subset_size]
    
    cov_orig = torch.cov(features_orig[:, indices].T)
    cov_approx = torch.cov(features_approx[:, indices].T)
    
    cov_frobenius_error = torch.norm(cov_orig - cov_approx, 'fro') / torch.norm(cov_orig, 'fro')
    cov_error = cov_frobenius_error.item()
    
    if verbose:
        print(f"\n3. Covariance matrix comparison (subset of {subset_size} features):")
        print(f"   Frobenius norm relative error: {cov_error * 100:.2f}%")
    
    # Test 4: Distribution shape comparison via histograms
    orig_flat = features_orig.flatten().cpu().numpy()
    approx_flat = features_approx.flatten().cpu().numpy()
    
    # Compute histogram comparison (L1 distance)
    bins = np.linspace(min(orig_flat.min(), approx_flat.min()), 
                       max(orig_flat.max(), approx_flat.max()), 100)
    hist_orig, _ = np.histogram(orig_flat, bins=bins, density=True)
    hist_approx, _ = np.histogram(approx_flat, bins=bins, density=True)
    
    l1_distance = np.sum(np.abs(hist_orig - hist_approx)) * (bins[1] - bins[0])
    
    if verbose:
        print(f"\n4. Distribution shape comparison:")
        print(f"   L1 distance between histograms: {l1_distance:.6f}")
    
    return {
        'mean_error': mean_error,
        'var_error': var_error, 
        'cov_error': cov_error,
        'dist_error': l1_distance
    }


# ============================================================================
# Activation Functions
# ============================================================================

def relu_activation(x: torch.Tensor) -> torch.Tensor:
    """ReLU activation function."""
    return F.relu(x)


def sigmoid_activation(x: torch.Tensor) -> torch.Tensor:
    """Sigmoid activation function."""
    return torch.sigmoid(x)


def tanh_activation(x: torch.Tensor) -> torch.Tensor:
    """Tanh activation function."""
    return torch.tanh(x)


def leaky_relu_activation(x: torch.Tensor, negative_slope: float = 0.1) -> torch.Tensor:
    """Leaky ReLU activation function."""
    return F.leaky_relu(x, negative_slope=negative_slope)


def gelu_activation(x: torch.Tensor) -> torch.Tensor:
    """GELU activation function."""
    return F.gelu(x)


def elu_activation(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """ELU activation function."""
    return F.elu(x, alpha=alpha)


def swish_activation(x: torch.Tensor) -> torch.Tensor:
    """Swish (SiLU) activation function."""
    return F.silu(x)


# Dictionary of available activation functions
ACTIVATION_FUNCTIONS = {
    'relu': relu_activation,
    'sigmoid': sigmoid_activation,
    'tanh': tanh_activation,
    'leaky_relu': leaky_relu_activation,
    'gelu': gelu_activation,
    'elu': elu_activation,
    'swish': swish_activation
}


# ============================================================================
# High-level interface functions
# ============================================================================

def compare_random_features(X: torch.Tensor,
                           F: torch.Tensor,
                           activation_name: str,
                           device: str = "cpu",
                           verbose: bool = True) -> Dict:
    """
    High-level function to compare original and approximated random features.
    
    Args:
        X: Input data (n_samples, input_dim)
        F: Random feature matrix (feature_dim, input_dim)
        activation_name: Name of activation function (from ACTIVATION_FUNCTIONS)
        device: Device to run computations on
        verbose: Whether to print results
        
    Returns:
        Dictionary containing moments, features, and validation results
    """
    if activation_name not in ACTIVATION_FUNCTIONS:
        raise ValueError(f"Unknown activation function: {activation_name}. "
                        f"Available: {list(ACTIVATION_FUNCTIONS.keys())}")
    
    activation_func = ACTIVATION_FUNCTIONS[activation_name]
    
    # Move tensors to device
    X = X.to(device)
    F = F.to(device)
    
    # Compute moments
    mu_0, mu_1, mu_star = compute_moments(activation_func, device=device)
    
    if verbose:
        print(f"=== {activation_name.upper()} Activation Analysis ===")
        print(f"Moments: μ₀={mu_0:.6f}, μ₁={mu_1:.6f}, μ*={mu_star:.6f}")
    
    # Generate features
    features_orig = random_feature_original(X, F, activation_func)
    features_approx = random_feature_approximation(X, F, mu_0, mu_1, mu_star)
    
    # Validate equivalence
    validation_results = validate_equivalence(features_orig, features_approx, verbose=verbose)
    
    return {
        'activation_name': activation_name,
        'moments': (mu_0, mu_1, mu_star),
        'features_original': features_orig,
        'features_approximated': features_approx,
        'validation': validation_results
    }


def batch_activation_comparison(X: torch.Tensor,
                               F: torch.Tensor,
                               activation_names: Optional[list] = None,
                               device: str = "cpu") -> Dict:
    """
    Compare multiple activation functions at once.
    
    Args:
        X: Input data (n_samples, input_dim)
        F: Random feature matrix (feature_dim, input_dim)
        activation_names: List of activation function names to test
        device: Device to run computations on
        
    Returns:
        Dictionary with results for each activation function
    """
    if activation_names is None:
        activation_names = list(ACTIVATION_FUNCTIONS.keys())
    
    results = {}
    
    print("=" * 80)
    print("BATCH ACTIVATION FUNCTION COMPARISON")
    print("=" * 80)
    
    for name in activation_names:
        print(f"\n{'-' * 40}")
        result = compare_random_features(X, F, name, device=device, verbose=True)
        results[name] = result
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Activation':<12} | {'μ₀':<8} {'μ₁':<8} {'μ*':<8} | {'Mean Err':<9} {'Dist Err':<9}")
    print("-" * 80)
    
    for name, data in results.items():
        mu_0, mu_1, mu_star = data['moments']
        val = data['validation']
        print(f"{name:<12} | {mu_0:8.4f} {mu_1:8.4f} {mu_star:8.4f} | "
              f"{val['mean_error']:9.6f} {val['dist_error']:9.6f}")
    
    return results