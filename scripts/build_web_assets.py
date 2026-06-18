"""
Build web assets for the project page from the local figure store.

Reads the OneDrive figure store, resizes/optimizes images into docs/assets/,
and emits docs/data/manifest.json + a representative spectrum for the kappa widget.

Usage:
    python scripts/build_web_assets.py

Override the source store with env var RMT_FIG_STORE.
"""
import os
import re
import json
import glob
from pathlib import Path

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent.parent
DOCS = HERE / "docs"
ASSETS = DOCS / "assets"
DATA = DOCS / "data"

STORE = Path(os.environ.get(
    "RMT_FIG_STORE",
    "/Users/binxuwang/Library/CloudStorage/OneDrive-HarvardUniversity/"
    "DiffusionRMT_consistency/Figures",
))

# paper figs (polished composites) live in the writing repo
PAPER = Path(os.environ.get(
    "RMT_PAPER_DIR",
    "/Users/binxuwang/Documents/Writings/ICML2026_DiffusionRMTConsistency",
))

DATASETS = [
    "FFHQ64", "FFHQ32", "CIFAR", "CIFAR100", "AFHQ32",
    "LSUNchurch32", "LSUNchurch64", "LSUNbedroom32", "LSUNbedroom64",
]

DATASET_LABELS = {
    "FFHQ64": "FFHQ 64×64 (faces)",
    "FFHQ32": "FFHQ 32×32 (faces)",
    "CIFAR": "CIFAR-10 32×32",
    "CIFAR100": "CIFAR-100 32×32",
    "AFHQ32": "AFHQ 32×32 (animals)",
    "LSUNchurch32": "LSUN Church 32×32",
    "LSUNchurch64": "LSUN Church 64×64",
    "LSUNbedroom32": "LSUN Bedroom 32×32",
    "LSUNbedroom64": "LSUN Bedroom 64×64",
}


def save_web(src, dst, max_w=1000, quality=88):
    """Resize an image to max width and save as optimized JPEG/PNG."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src).convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    im.save(dst, quality=quality, optimize=True)
    return dst.relative_to(DOCS).as_posix()


def save_web_img(im, dst, max_w=520, quality=88):
    """Save an already-loaded PIL image, resized to max width."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = im.convert("RGB")
    if im.width > max_w:
        h = round(im.height * max_w / im.width)
        im = im.resize((max_w, h), Image.LANCZOS)
    im.save(dst, quality=quality, optimize=True)
    return dst.relative_to(DOCS).as_posix()


# ----------------------------------------------------------------------------
def build_seed_explorer():
    """Per-seed CNN/DiT/Linear/closest comparison composites for each dataset."""
    grid_root = STORE / "Figure_motivation_visual_sample_grid"
    manifest = {}
    for ds in DATASETS:
        d = grid_root / ds
        if not d.exists():
            print(f"  [skip] {ds}: no folder")
            continue
        pat = re.compile(
            rf"sample_grid_.*closest_split12_cmp_{ds}_(\d+)_(\d+)\.png$")
        found = {}  # seed -> (n, path)
        for p in sorted(d.glob("*.png")):
            m = pat.search(p.name)
            if not m:
                continue
            n, seed = int(m.group(1)), int(m.group(2))
            # keep the largest dataset-size variant per seed
            if seed not in found or n > found[seed][0]:
                found[seed] = (n, p)
        if not found:
            print(f"  [skip] {ds}: no matching grids")
            continue
        seeds = sorted(found)
        n_used = found[seeds[0]][0]
        has_dit = "CNN_DiT_Linear" in found[seeds[0]][1].name
        for seed in seeds:
            rel = save_web(found[seed][1],
                           ASSETS / "seeds" / ds / f"seed_{seed}.jpg",
                           max_w=900)
        manifest[ds] = {
            "label": DATASET_LABELS[ds],
            "n": n_used,
            "seeds": seeds,
            "has_dit": has_dit,
            "path": f"assets/seeds/{ds}/seed_{{seed}}.jpg",
        }
        print(f"  {ds}: {len(seeds)} seeds (n={n_used}, dit={has_dit})")
    return manifest


def build_size_slider_facegrids():
    """FFHQ32/64 size grids extracted from the combined multi-panel validation
    figures (CNN column, split-1 top row / split-2 bottom row)."""
    root = STORE / "DNN_validation"
    # dataset -> CNN-column x-bounds (fraction). y: split1 / split2 rows.
    cfg = {
        "FFHQ64": (0.675, 0.995),   # CNN is the rightmost of DiT_P2/DiT_P4/CNN
        "FFHQ32": (0.368, 0.652),   # CNN is the middle of DiT/CNN/MLP
    }
    yrows = {1: (0.095, 0.495), 2: (0.555, 0.955)}
    manifest = {}
    for ds, (x0, x1) in cfg.items():
        pat = re.compile(rf"{ds}_final_samples_images_(\d+)\.png$")
        ns = []
        for p in sorted((root / ds).glob(f"{ds}_final_samples_images_*.png")):
            m = pat.search(p.name)
            if not m:
                continue
            n = int(m.group(1))
            im = Image.open(p).convert("RGB")
            W, H = im.size
            for split, (y0, y1) in yrows.items():
                panel = im.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
                save_web_img(panel, ASSETS / "sizes" / ds / f"n_{n}_split{split}.jpg", max_w=520)
            ns.append(n)
        if not ns:
            print(f"  [skip] facegrid {ds}: none")
            continue
        ns = sorted(ns)
        manifest[ds] = {
            "label": DATASET_LABELS[ds],
            "arch": "UNet_CNN",
            "sizes": ns,
            "splits": [1, 2],
            "path": f"assets/sizes/{ds}/n_{{n}}_split{{split}}.jpg",
        }
        print(f"  {ds}: sizes {ns} (split 1 & 2, from validation grid)")
    return manifest


def build_size_slider():
    """Memorization -> renormalization: split-1 AND split-2 sample grids across sizes.

    NOTE: FFHQ/AFHQ are intentionally NOT sourced from the DNN_validation combined
    grids — those are sorted by nearest-neighbour distance, not seed, so split-1 and
    split-2 cells are not seed-aligned. Add them here once seed-ordered montages exist
    in DNN_final_samples following the same naming convention (see README/build notes).
    """
    manifest = {}
    src_root = STORE / "DNN_final_samples"
    # dataset -> arch token used in filenames
    targets = {
        "CIFAR": "UNet_CNN",
        "CIFAR100": "UNet_CNN",
        "LSUNchurch64": "UNet_CNN",
        "LSUNchurch32": "UNet_CNN",
        "LSUNbedroom64": "UNet_CNN",
        "LSUNbedroom32": "UNet_CNN",
    }
    for ds, arch in targets.items():
        # size -> {split: path}, keep only sizes present in BOTH splits
        sizes = {}
        for split in (1, 2):
            pat = re.compile(rf"{ds}_(\d+)_{arch}_EDM_DSM_split{split}.*samples.*\.png$")
            for p in sorted(src_root.glob(f"{ds}_*split{split}*.png")):
                m = pat.search(p.name)
                if not m:
                    continue
                n = int(m.group(1))
                sizes.setdefault(n, {})[split] = p
        ns = sorted(n for n, sp in sizes.items() if 1 in sp and 2 in sp)
        if not ns:
            print(f"  [skip] size-slider {ds}: none")
            continue
        for n in ns:
            for split in (1, 2):
                save_web(sizes[n][split],
                         ASSETS / "sizes" / ds / f"n_{n}_split{split}.jpg", max_w=520)
        manifest[ds] = {
            "label": DATASET_LABELS[ds],
            "arch": arch,
            "sizes": ns,
            "splits": [1, 2],
            "path": f"assets/sizes/{ds}/n_{{n}}_split{{split}}.jpg",
        }
        print(f"  {ds}: sizes {ns} (split 1 & 2)")
    return manifest


def build_static_figs():
    """Copy polished composite figures used in theory / validation sections."""
    figs = {
        "motivation": PAPER / "figs" / "Figure_consistency_motivation-01.png",
        "noise_renorm": PAPER / "figs" / "Figure_noise_renormalization_eff-02.png",
        "denoiser_var": PAPER / "figs" / "Figure_denoiser_variance-01.png",
        "sampling_exp_var": PAPER / "figs" / "Figure_generative_mapping_exp_variance-03.png",
        "dnn_validation": PAPER / "figs" / "Figure_DNN_valid_exp_var.png",
    }
    out = {}
    for key, src in figs.items():
        if not src.exists():
            print(f"  [skip] fig {key}: missing {src}")
            continue
        rel = save_web(src, ASSETS / "theory" / f"{key}.jpg", max_w=1400, quality=90)
        out[key] = rel
        print(f"  {key} -> {rel}")
    return out


def build_counterfactual():
    """Counterfactual 'moments differ -> consistency breaks' figures."""
    base = STORE / "dataset_mean_cov_mismatch_splits" / "FFHQ32"
    srcs = {
        "cf_kde": base / "FFHQ32_PC2_top_bottom_mid_kde_scores.png",
        "cf_heatmap": base / "FFHQ32_DiT_P2_N3000_splits_mean_cov_manip_cmp.png",
    }
    out = {}
    for key, src in srcs.items():
        if not src.exists():
            print(f"  [skip] {key}: missing {src}")
            continue
        out[key] = save_web(src, ASSETS / "counterfactual" / f"{key}.jpg",
                            max_w=1300, quality=90)
        print(f"  {key} -> {out[key]}")
    return out


def build_spectrum(d=3072, n_eig=512, alpha=1.6, floor=1e-4, seed=0):
    """Representative heavy-tailed natural-image covariance spectrum for the
    kappa(sigma^2) widget. Power-law decay with a small noise floor, normalised
    so the mean eigenvalue is 1. Labelled as 'representative' on the page.
    """
    k = np.arange(1, n_eig + 1)
    lam = k.astype(float) ** (-alpha) + floor
    lam = lam / lam.mean()  # mean eigenvalue = 1
    # weights so trace is preserved when subsampling (each eigenvalue equal mass)
    weights = np.ones_like(lam) / len(lam)
    return {
        "label": "Representative natural-image spectrum (power-law, α=%.1f)" % alpha,
        "d": d,
        "eigenvalues": [float(x) for x in lam],
        "weights": [float(x) for x in weights],
    }


def main():
    print("Seed explorer:")
    seeds = build_seed_explorer()
    print("Size slider:")
    sizes = build_size_slider()
    print("Static figures:")
    figs = build_static_figs()
    print("Counterfactual figures:")
    figs.update(build_counterfactual())
    print("Spectrum:")
    spectrum = build_spectrum()
    print(f"  {len(spectrum['eigenvalues'])} eigenvalues")

    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {
        "datasets_order": [d for d in DATASETS if d in seeds],
        "seed_explorer": seeds,
        "size_slider": sizes,
        "figures": figs,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (DATA / "spectrum.json").write_text(json.dumps(spectrum))
    print(f"\nWrote {DATA/'manifest.json'} and {DATA/'spectrum.json'}")


if __name__ == "__main__":
    main()
