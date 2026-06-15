"""Checkpoint/resume helpers for long cluster inference runs."""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, Optional


def _parse_bool(value: Optional[str], default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def is_resume_enabled(default: bool = False) -> bool:
    return _parse_bool(os.environ.get("ROR_RESUME"), default=default)


def get_checkpoint_every(default: int = 50) -> int:
    raw = os.environ.get("ROR_CHECKPOINT_EVERY")
    if raw is None or not str(raw).strip():
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def get_snapshot_every(default: int = 50) -> int:
    """
    Snapshot cadence for partial result payloads.

    If ROR_SNAPSHOT_EVERY is unset/invalid, falls back to `default` which can
    be aligned with checkpoint cadence.
    """
    raw = os.environ.get("ROR_SNAPSHOT_EVERY")
    if raw is None or not str(raw).strip():
        return max(1, int(default))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return max(1, int(default))


def load_checkpoint(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _save_atomic_json(path: str, payload: Dict[str, Any], prefix: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_checkpoint(path: str, payload: Dict[str, Any]) -> None:
    _save_atomic_json(path=path, payload=payload, prefix=".checkpoint_")


def save_result_snapshot(path: str, payload: Dict[str, Any]) -> None:
    _save_atomic_json(path=path, payload=payload, prefix=".snapshot_")


def clear_checkpoint(path: str) -> None:
    if path and os.path.exists(path):
        os.remove(path)
