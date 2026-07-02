"""Deep SVDD (one-class) anomaly scorer for DGA risk.

Deep Support Vector Data Description (Ruff et al., 2018): a network phi maps each
sample to a representation space and is trained to pull the data into a minimum-volume
hypersphere around a fixed centre c; the anomaly score is the squared distance to c.

Standard anti-collapse choices: no bias terms, unbounded (LeakyReLU) activations, the
centre c fixed to the initial mean output (nudged away from 0), and weight decay.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


class _Net(nn.Module):
    def __init__(self, in_dim: int, hidden=(16, 8), rep_dim: int = 4):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h, bias=False), nn.LeakyReLU()]
            d = h
        layers += [nn.Linear(d, rep_dim, bias=False)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def deep_svdd_scores(X: np.ndarray, rep_dim: int = 4, hidden=(16, 8), epochs: int = 80,
                     lr: float = 1e-3, weight_decay: float = 1e-5, seed: int = 42) -> np.ndarray:
    """Train Deep SVDD on X and return the per-sample anomaly score (higher = more anomalous)."""
    torch.manual_seed(seed)
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    net = _Net(Xt.shape[1], tuple(hidden), rep_dim)

    with torch.no_grad():                       # fixed centre = initial mean output
        c = net(Xt).mean(0)
        c[c.abs() < 1e-6] = 1e-6                 # avoid a trivial all-zero centre

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt) - c) ** 2).sum(dim=1).mean()
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        return ((net(Xt) - c) ** 2).sum(dim=1).cpu().numpy()
