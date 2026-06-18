"""Helpers for analyzing deep-network samples.

Nearest-neighbour lookups (for the memorization checks) and parsing of
experiment directory names into a tidy DataFrame.
"""
import os
from glob import glob
from os.path import join

import torch
import pandas as pd

def find_closest_k_indices(Xtsr, probe, k=5):
    """
    Find the k closest indices to a probe tensor in the dataset.
    
    Args:
        Xtsr: Dataset tensor of shape (N, C, H, W)
        probe: Probe tensor of shape (C, H, W) or (1, C, H, W)
        k: Number of closest indices to return
    
    Returns:
        indices: Tensor of k closest indices
    """
    # Ensure probe has batch dimension
    if probe.dim() == 3:
        probe = probe.unsqueeze(0)
    # Flatten tensors for distance computation
    Xtsr_flat = Xtsr.flatten(start_dim=1)  # (N, C*H*W)
    probe_flat = probe.flatten(start_dim=1)  # (1, C*H*W)
    # Compute L2 distances
    distances = torch.cdist(probe_flat, Xtsr_flat, p=2).squeeze(0)  # (N,)
    # Find k closest indices
    topk_distances, indices = torch.topk(distances, k, largest=False)
    return indices, topk_distances


def find_closest_k_indices_batch(Xtsr, probes, k=5):
    """
    Find the k closest indices to multiple probe tensors in the dataset.
    
    Args:
        Xtsr: Dataset tensor of shape (N, C, H, W)
        probes: Probe tensors of shape (B, C, H, W) where B is the number of probes
        k: Number of closest indices to return for each probe
    
    Returns:
        indices: Tensor of shape (B, k) with k closest indices for each probe
    """
    # Flatten tensors for distance computation
    Xtsr_flat = Xtsr.flatten(start_dim=1)  # (N, C*H*W)
    probes_flat = probes.flatten(start_dim=1)  # (B, C*H*W)
    # Compute pairwise L2 distances: (B, N)
    # Using cdist for efficient distance computation
    distances = torch.cdist(probes_flat, Xtsr_flat, p=2)  # (B, N)
    # Find k closest indices for each probe
    topk_distances, indices = torch.topk(distances, k, dim=1, largest=False)  # (B, k)
    return indices, topk_distances



def parse_experiments(exproot, exp_patterns, verbose=False):
    """
    Find and parse experiment directories based on patterns.
    
    Args:
        exproot (str): Root directory to search for experiments
        exp_patterns (list): List of glob patterns to match experiment directories
    
    Returns:
        pd.DataFrame: DataFrame containing parsed experiment information
    """
    exp_dirs = []
    for pattern in exp_patterns:
        exp_dirs.extend(glob(join(exproot, pattern)))
    exp_dirs = sorted(exp_dirs)
    print(f"Found {len(exp_dirs)} experiments:")
    if verbose:
        for exp_dir in exp_dirs:
            print(os.path.basename(exp_dir))
    
    # Parse experiment names and create a simple dataframe
    exp_names = [os.path.basename(exp_dir) for exp_dir in exp_dirs]
    # Parse experiment names
    parsed_data = []
    for exp_name in exp_names:
        # Extract dataset name
        if exp_name.startswith('FFHQ32_'):
            dataset = 'FFHQ32'
            resolution = 32
        elif exp_name.startswith('AFHQ32_'):
            dataset = 'AFHQ32'
            resolution = 32
        elif exp_name.startswith('CIFAR_'):
            dataset = 'CIFAR'
            resolution = 32
        elif exp_name.startswith('FFHQ64'):
            dataset = 'FFHQ64'
            resolution = 64
        elif exp_name.startswith('CIFAR100_'):
            dataset = 'CIFAR100'
            resolution = 32
        elif exp_name.startswith('LSUNchurch64_'):
            dataset = 'LSUNchurch64'
            resolution = 64
        elif exp_name.startswith('LSUNchurch32_'):
            dataset = 'LSUNchurch32'
            resolution = 32
        elif exp_name.startswith('LSUNbedroom64_'):
            dataset = 'LSUNbedroom64'
            resolution = 64
        elif exp_name.startswith('LSUNbedroom32_'):
            dataset = 'LSUNbedroom32'
            resolution = 32
        else:
            dataset = 'Unknown'
            resolution = None
            print(f"Unknown dataset: {exp_name}")
        
        dataset_size = int(exp_name.split('_')[1])
        # Extract model type
        if 'DiT' in exp_name:
            model_type = 'DiT'
            if "P2" in exp_name:
                model_type = 'DiT_P2'
            elif "P4" in exp_name:
                model_type = 'DiT_P4'
        elif 'UNet_CNN' in exp_name:
            model_type = 'CNN'
        elif 'UNet_MLP' in exp_name:
            model_type = 'MLP'
        else:
            model_type = 'Unknown'
        
        # Extract split
        if 'split1' in exp_name:
            split = 1
        elif 'split2' in exp_name:
            split = 2
        else:
            split = None
        
        parsed_data.append({
            'exp_name': exp_name,
            'dataset': dataset,
            'resolution': resolution,
            'dataset_size': dataset_size,
            'model_type': model_type,
            'split': split
        })

    # Create DataFrame
    df_expnames = pd.DataFrame(parsed_data)
    print(f"Created dataframe with {len(df_expnames)} experiments")
    print(df_expnames)
    return df_expnames
