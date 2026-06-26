"""Load dataset/path configuration from a gitignored ``config.yaml``.

Machine- and dataset-specific paths (e.g. absolute data locations) live in
``config.yaml`` so they stay out of the committed code. See
``config.example.yaml`` for the schema.
"""
import os

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_dataset_config(key, config_path=None):
    """Return the config dict for dataset ``key`` from ``config.yaml``."""
    config_path = config_path or os.path.join(_REPO_ROOT, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"{config_path} not found; copy config.example.yaml to config.yaml "
            "and fill in your paths."
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}
    datasets = cfg.get("datasets", {})
    if key not in datasets:
        raise KeyError(
            f"dataset '{key}' not in {config_path}; available: {sorted(datasets)}"
        )
    return datasets[key]
