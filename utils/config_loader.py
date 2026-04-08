"""
Config loader — loads config.yaml once and returns as dict.

Single entry point for all configuration across the engine.
Every module receives config as a dict parameter rather than
loading it independently, ensuring consistency.
"""

import os
import yaml


_CONFIG_CACHE = None


def load_config(path: str = None) -> dict:
    """
    Load config.yaml from project root and return as dict.

    Uses module-level cache so the file is read at most once per
    process. Pass an explicit path to override (useful in tests).

    Args:
        path: Optional absolute path to config.yaml.

    Returns:
        Configuration dictionary.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and path is None:
        return _CONFIG_CACHE

    if path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "config.yaml")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    if path is None or "config.yaml" in path:
        _CONFIG_CACHE = config

    return config


def reset_cache():
    """Clear the config cache. Useful in tests."""
    global _CONFIG_CACHE
    _CONFIG_CACHE = None
