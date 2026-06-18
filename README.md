# A Random Matrix Theory Perspective on the Consistency of Diffusion Models

**ICML 2026 (Oral).** Binxu Wang, Jacob A. Zavatone-Veth, Cengiz Pehlevan — Harvard University.

[📄 Paper / arXiv](https://arxiv.org/abs/2602.02908) · [💻 Code](https://github.com/Animadversio/Diffusion_RMT_consistency)

Diffusion models trained on **disjoint** halves of a dataset produce nearly the **same image**
from the **same noise seed** — across data splits *and* architectures (UNet, DiT). We show this
*consistency* is largely a **linear** effect (shared Gaussian statistics already predict the output)
and develop a **random matrix theory** that predicts precisely *where* and *how much* independently
trained models disagree: finite data renormalize the noise scale `σ² ↦ κ(σ²)`,
over-shrinking low-variance modes, and cross-split disagreement factorizes into
**anisotropy × inhomogeneity × scaling**.

## Project page

The interactive project page lives in [`docs/`](docs/) (served via GitHub Pages):

- **Seed explorer** — compare split-1 / split-2 / linear-theory generations for any seed across 9 datasets.
- **Memorization → renormalization** — slide the training-set size to watch the transition.
- **Live κ(σ²) widget** — solve the Silverstein self-consistency equation in your browser.

### Build / preview locally

```bash
# regenerate web assets from the local figure store (set RMT_FIG_STORE if needed)
python scripts/build_web_assets.py
# preview
python -m http.server 8731 --directory docs   # → http://localhost:8731
```

### Deploy to GitHub Pages

Two supported paths (both work — pick one):

- **GitHub Actions (default):** `.github/workflows/pages.yml` publishes `docs/` on every push to
  `main`. Enable it once via **Settings → Pages → Source: GitHub Actions**.
- **Branch folder:** **Settings → Pages → Source: Deploy from a branch → `main` / `/docs`**.

The site uses only **relative paths** and ships a `docs/.nojekyll` file, so it works correctly at a
project-page subpath (`https://<user>.github.io/<repo>/`) with no Jekyll processing.

## Repository layout

```
docs/                         # interactive project page (GitHub Pages root)
scripts/build_web_assets.py   # builds docs/assets + manifest + spectrum from the figure store
rmt_core/   (planned)         # reusable RMT toolkit: κ-solver, deterministic equivalence, Wiener filter
notebooks/  (planned)         # curated example notebooks reproducing key figures
```

> The reusable RMT computing machinery (`rmt_core`) and curated notebooks are being
> documented for release; see the [code repo](https://github.com/Animadversio/Diffusion_RMT_consistency) in the meantime.

## Citation

```bibtex
@inproceedings{wang2026rmtconsistency,
  title     = {A Random Matrix Theory Perspective on the Consistency of Diffusion Models},
  author    = {Wang, Binxu and Zavatone-Veth, Jacob A. and Pehlevan, Cengiz},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
  note      = {Oral presentation},
  url       = {https://arxiv.org/abs/2602.02908}
}
```
