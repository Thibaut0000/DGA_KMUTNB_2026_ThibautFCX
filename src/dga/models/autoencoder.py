"""Configurable MLP Autoencoder (PyTorch) for compact DGA representations.

Small tabular data (~4.5k rows, <=7 features) — this trains in seconds on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(16, 8), latent_dim: int = 3,
                 activation: str = "relu", dropout: float = 0.0):
        super().__init__()
        act = _ACT[activation]
        enc, d = [], input_dim
        for h in hidden_dims:
            enc += [nn.Linear(d, h), act()]
            if dropout:
                enc += [nn.Dropout(dropout)]
            d = h
        enc += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec, d = [], latent_dim
        for h in reversed(hidden_dims):
            dec += [nn.Linear(d, h), act()]
            if dropout:
                dec += [nn.Dropout(dropout)]
            d = h
        dec += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        return self.encoder(t).cpu().numpy()

    @torch.no_grad()
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Per-sample mean squared reconstruction error (anomaly score)."""
        self.eval()
        t = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        recon, _ = self.forward(t)
        return ((recon - t) ** 2).mean(dim=1).cpu().numpy()


@dataclass
class TrainResult:
    model: Autoencoder
    history: dict = field(default_factory=dict)


def train_autoencoder(X: np.ndarray, cfg, *, verbose: bool = True) -> TrainResult:
    """Train with an early-stopping validation split. `cfg` = config.autoencoder."""
    torch.manual_seed(getattr(cfg, "seed", 42))
    X = np.asarray(X, dtype=np.float32)
    ds = TensorDataset(torch.from_numpy(X))
    n_val = max(1, int(len(ds) * cfg.val_split))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size)

    model = Autoencoder(X.shape[1], tuple(cfg.hidden_dims), cfg.latent_dim,
                        cfg.activation, cfg.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    history = {"train": [], "val": []}
    for epoch in range(cfg.epochs):
        model.train()
        tr = 0.0
        for (xb,) in train_dl:
            opt.zero_grad()
            recon, _ = model(xb)
            loss = loss_fn(recon, xb)
            loss.backward()
            opt.step()
            tr += loss.item() * len(xb)
        tr /= len(train_ds)

        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model(xb)[0], xb).item() * len(xb) for (xb,) in val_dl) / len(val_ds)
        history["train"].append(tr)
        history["val"].append(vl)

        if vl < best_val - 1e-6:
            best_val, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if verbose and (epoch % 20 == 0 or epoch == cfg.epochs - 1):
            print(f"epoch {epoch:3d}  train {tr:.4f}  val {vl:.4f}")
        if patience >= cfg.early_stopping_patience:
            if verbose:
                print(f"early stop @ epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history)
