"""Step 2 — build features, train the Autoencoder, save model + latent codes.

    python scripts/train_ae.py
Outputs:
    results/models/autoencoder.pt
    results/models/latent.npy        latent codes aligned to feature index
    results/models/feature_index.npy sample_ids retained
"""
import _bootstrap  # noqa: F401
import numpy as np
import torch

from dga.config import PROJECT_ROOT, load_config, set_seed
from dga import data as dga_data
from dga.preprocessing import build_feature_matrix
from dga.models.autoencoder import train_autoencoder

cfg = load_config()


def main():
    set_seed(cfg.seed)
    df = dga_data.load_clean()
    fm = build_feature_matrix(df, cfg)
    print(f"feature matrix: {fm.X.shape}  features={fm.features}")

    res = train_autoencoder(fm.X, cfg.autoencoder)
    Z = res.model.encode(fm.X)

    mdir = PROJECT_ROOT / "results" / "models"
    mdir.mkdir(parents=True, exist_ok=True)
    torch.save(res.model.state_dict(), mdir / "autoencoder.pt")
    np.save(mdir / "latent.npy", Z)
    np.save(mdir / "feature_index.npy", fm.index.to_numpy())
    print("saved latent codes", Z.shape, "->", mdir / "latent.npy")


if __name__ == "__main__":
    main()
