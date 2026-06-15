"""Helpers for configuring cluster-local cache directories."""

from __future__ import annotations

import os
from typing import Dict, Optional


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def resolve_cache_root(
    cache_root: Optional[str] = None,
    use_tmpdir_if_available: bool = False,
) -> Optional[str]:
    """Resolve the cache root path from explicit input and environment hints."""
    root = cache_root
    if root is None or not str(root).strip():
        if use_tmpdir_if_available:
            tmpdir = os.environ.get("TMPDIR")
            if tmpdir and str(tmpdir).strip():
                root = os.path.join(tmpdir, "ror-cache")
        if root is None or not str(root).strip():
            env_root = os.environ.get("ROR_CLUSTER_CACHE_ROOT")
            if env_root and str(env_root).strip():
                root = env_root
    if root is None or not str(root).strip():
        return None
    return os.path.abspath(os.path.expanduser(str(root).strip()))


def configure_cluster_cache_env(
    cache_root: Optional[str] = None,
    use_tmpdir_if_available: bool = False,
) -> Dict[str, str]:
    """
    Configure cache-related environment variables for cluster runs.

    Returns a dictionary of env vars that were set.
    """
    root = resolve_cache_root(
        cache_root=cache_root,
        use_tmpdir_if_available=use_tmpdir_if_available,
    )
    if root is None:
        return {}

    hf_home = _ensure_dir(os.path.join(root, "hf"))
    env_map = {
        "ROR_CLUSTER_CACHE_ROOT": root,
        "HF_HOME": hf_home,
        "HF_HUB_CACHE": _ensure_dir(os.path.join(hf_home, "hub")),
        "TRANSFORMERS_CACHE": _ensure_dir(os.path.join(hf_home, "transformers")),
        "XDG_CACHE_HOME": _ensure_dir(os.path.join(root, "xdg")),
        "TRITON_CACHE_DIR": _ensure_dir(os.path.join(root, "triton")),
        "TORCH_HOME": _ensure_dir(os.path.join(root, "torch")),
        "CUDA_CACHE_PATH": _ensure_dir(os.path.join(root, "cuda")),
    }

    for key, value in env_map.items():
        os.environ[key] = value
    return env_map
