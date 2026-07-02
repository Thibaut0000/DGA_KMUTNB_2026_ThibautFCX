"""Tiny YAML-backed configuration with attribute access.

    from dga.config import load_config, set_seed
    cfg = load_config()                 # loads config/default.yaml
    print(cfg.autoencoder.latent_dim)   # -> 3
"""
from __future__ import annotations

import random
from pathlib import Path

import yaml

# .../IR Bangkok  (repo root = two levels above this file: src/dga/config.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict that also supports `cfg.section.key` attribute access."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(key) from exc
        return Config(value) if isinstance(value, dict) else value

    def __setattr__(self, key, value):
        self[key] = value


def load_config(path: str | Path | None = None) -> Config:
    """Load a YAML config (defaults to config/default.yaml)."""
    path = Path(path) if path else PROJECT_ROOT / "config" / "default.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return Config(yaml.safe_load(fh))


def resolve(path_like: str | Path) -> Path:
    """Resolve a path relative to the project root (absolute paths pass through)."""
    p = Path(path_like)
    return p if p.is_absolute() else PROJECT_ROOT / p


def set_seed(seed: int = 42) -> None:
    """Seed Python / NumPy / PyTorch for reproducible experiments."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
