"""SD-CAE: Severity-Disentangled Compositional Autoencoder (PyTorch).

The encoder sees only the CLR composition, so the latent z is scale-invariant by
construction (same fault type at any gassing level -> same z). An optional
adversary tries to recover the magnitude m from z through a gradient-reversal
layer, pushing z to be statistically independent of severity; lambda controls
the strength (0 disables, and the adversary then acts only as an m-probe).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float):
    return _GradReverse.apply(x, lambd)


def _mlp(dims, act, dropout):
    layers, d = [], dims[0]
    for h in dims[1:-1]:
        layers += [nn.Linear(d, h), act()]
        if dropout:
            layers += [nn.Dropout(dropout)]
        d = h
    layers += [nn.Linear(d, dims[-1])]
    return nn.Sequential(*layers)


class SDCAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=(16, 8), latent_dim=2,
                 activation="relu", dropout=0.0, adv_hidden=16):
        super().__init__()
        act = _ACT[activation]
        self.encoder = _mlp([input_dim, *hidden_dims, latent_dim], act, dropout)
        self.decoder = _mlp([latent_dim, *reversed(hidden_dims), input_dim], act, dropout)
        # adversary: latent -> magnitude scalar
        self.adversary = _mlp([latent_dim, adv_hidden, 1], act, 0.0)

    def forward(self, c):
        z = self.encoder(c)
        return self.decoder(z), z

    @torch.no_grad()
    def encode(self, C: np.ndarray) -> np.ndarray:
        self.eval()
        t = torch.as_tensor(np.asarray(C), dtype=torch.float32)
        return self.encoder(t).cpu().numpy()

    @torch.no_grad()
    def reconstruction_error(self, C: np.ndarray) -> np.ndarray:
        """Per-sample composition reconstruction error (anomaly score, type-novelty)."""
        self.eval()
        t = torch.as_tensor(np.asarray(C), dtype=torch.float32)
        recon, _ = self.forward(t)
        return ((recon - t) ** 2).mean(dim=1).cpu().numpy()


@dataclass
class TrainResult:
    model: SDCAE
    history: dict = field(default_factory=dict)


def train_sdcae(C: np.ndarray, m: np.ndarray, cfg, *, verbose: bool = True) -> TrainResult:
    """Train SD-CAE. `cfg` = config.sdcae. `C` = CLR composition, `m` = magnitude."""
    seed = int(getattr(cfg, "seed", 42))
    torch.manual_seed(seed)
    C = np.asarray(C, dtype=np.float32)
    m = np.asarray(m, dtype=np.float32).reshape(-1, 1)
    lambd = float(cfg.lambda_adv)

    ds = TensorDataset(torch.from_numpy(C), torch.from_numpy(m))
    n_val = max(1, int(len(ds) * cfg.val_split))
    tr_ds, va_ds = random_split(ds, [len(ds) - n_val, n_val],
                                generator=torch.Generator().manual_seed(seed))
    tr_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=cfg.batch_size)

    model = SDCAE(C.shape[1], tuple(cfg.hidden_dims), cfg.latent_dim,
                  cfg.activation, cfg.dropout, cfg.adv_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    mse = nn.MSELoss()

    best_val, best_state, patience = float("inf"), None, 0
    history = {"train": [], "val": [], "rec": [], "adv": []}
    for epoch in range(cfg.epochs):
        model.train()
        tr = 0.0
        for cb, mb in tr_dl:
            opt.zero_grad()
            recon, z = model(cb)
            l_rec = mse(recon, cb)
            m_hat = model.adversary(grad_reverse(z, lambd))
            l_adv = mse(m_hat, mb)
            loss = l_rec + l_adv                # GRL flips l_adv's sign for the encoder
            loss.backward()
            opt.step()
            tr += l_rec.item() * len(cb)        # track reconstruction (the AE objective)
        tr /= len(tr_ds)

        model.eval()
        with torch.no_grad():
            vl = sum(mse(model(cb)[0], cb).item() * len(cb) for cb, _ in va_dl) / len(va_ds)
        history["train"].append(tr)
        history["val"].append(vl)

        if vl < best_val - 1e-6:
            best_val, best_state, patience = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if verbose and (epoch % 40 == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:3d}  rec(train) {tr:.4f}  rec(val) {vl:.4f}")
        if patience >= cfg.early_stopping_patience:
            if verbose:
                print(f"  early stop @ epoch {epoch} (best val {best_val:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(model=model, history=history)
