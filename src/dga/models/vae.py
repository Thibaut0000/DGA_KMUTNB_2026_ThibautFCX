"""Variational Autoencoder (beta-VAE) for DGA — probabilistic latent space.

ELBO = reconstruction (MSE) + beta * KL(q(z|x) || N(0, I)).
Set beta>1 in config for stronger disentanglement, <1 to favour reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(16, 8), latent_dim: int = 3):
        super().__init__()
        enc, d = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.encoder = nn.Sequential(*enc)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

        dec, d = [], latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    @torch.no_grad()
    def encode_mean(self, X: np.ndarray) -> np.ndarray:
        """Latent codes = posterior mean (deterministic, good for clustering)."""
        self.eval()
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        mu, _ = self.encode(t)
        return mu.cpu().numpy()

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        recon, _, _ = self.forward(t)
        return ((recon - t) ** 2).mean(dim=1).cpu().numpy()


def vae_loss(recon, x, mu, logvar, beta: float = 1.0):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl, recon_loss, kl


@dataclass
class VAEResult:
    model: VAE
    history: dict = field(default_factory=dict)


def train_vae(X: np.ndarray, cfg, *, verbose: bool = True) -> VAEResult:
    """`cfg` = config.vae."""
    torch.manual_seed(42)
    X = np.asarray(X, dtype=np.float32)
    dl = DataLoader(TensorDataset(torch.from_numpy(X)),
                    batch_size=cfg.batch_size, shuffle=True)
    model = VAE(X.shape[1], tuple(cfg.hidden_dims), cfg.latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"loss": [], "recon": [], "kl": []}
    for epoch in range(cfg.epochs):
        model.train()
        agg = np.zeros(3)
        for (xb,) in dl:
            opt.zero_grad()
            recon, mu, logvar = model(xb)
            loss, rl, kl = vae_loss(recon, xb, mu, logvar, cfg.beta)
            loss.backward()
            opt.step()
            agg += np.array([loss.item(), rl.item(), kl.item()]) * len(xb)
        agg /= len(X)
        history["loss"].append(agg[0])
        history["recon"].append(agg[1])
        history["kl"].append(agg[2])
        if verbose and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:3d}  elbo {agg[0]:.4f}  recon {agg[1]:.4f}  kl {agg[2]:.4f}")
    return VAEResult(model=model, history=history)
