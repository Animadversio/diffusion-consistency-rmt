"""Linear (Gaussian) denoiser and Wiener-filter sampling map.

Closed-form optimal denoiser and probability-flow sampling map for Gaussian
data, used as the linear-theory predictor throughout the paper.
"""
import torch


def dnoised_X(x, Xmean, sample_cov, sigma2):
    """One-step optimal linear denoiser: mu + Sigma (Sigma + sigma^2 I)^{-1} (x - mu)."""
    return x + sigma2 * (Xmean - x) @ torch.inverse(sample_cov + torch.eye(sample_cov.shape[0], device=x.device) * sigma2)


def wiener_gen_X(x, Xmean, wiener_matrix, sigmaT):
    """Map initial noise x (~ N(0, sigmaT^2 I)) to a generated sample via the Wiener matrix."""
    if x.dim() == 1:
        # Single vector case
        return Xmean + wiener_matrix @ (x * sigmaT - Xmean)
    else:
        # Batched vector case - x should be shape (batch_size, ndim)
        return Xmean[None, :] + (x * sigmaT - Xmean[None, :]) @ wiener_matrix.T


def build_wiener_matrix(eigvals, eigvecs, sigmaT=80.0, sigma0=0.0, EPS=1E-16, clip=True):
    """Build the Wiener filter matrix U diag(sqrt((lam+sigma0^2)/(lam+sigmaT^2))) U^T."""
    if clip:
        eigvals = torch.clamp(eigvals, min=EPS)
    scaling = ((eigvals + sigma0**2) / (eigvals + sigmaT**2)).sqrt()
    return eigvecs @ torch.diag(scaling) @ eigvecs.T


def tsr2img(x, imgshape=(3, 32, 32)):
    """Reshape a flattened [-1, 1] image tensor to an HxWxC numpy array in [0, 1]."""
    return (x.detach().cpu().view(imgshape).permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
