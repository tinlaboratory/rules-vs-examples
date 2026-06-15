"""Dataset loading helpers shared by task runners."""

from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence


def get_data_root(workspace_root: str) -> str:
    """Return the dataset root.

    By default the code looks for ``data/`` inside the code repository. When
    using the separate Hugging Face dataset checkout, set
    ``RULES_VS_EXAMPLES_DATA_DIR`` to that checkout's ``data`` directory.
    """
    configured = os.environ.get("RULES_VS_EXAMPLES_DATA_DIR")
    if configured and configured.strip():
        return os.path.abspath(os.path.expanduser(configured.strip()))
    local_data = os.path.join(workspace_root, "data")
    if os.path.isdir(local_data):
        return local_data
    sibling_data = os.path.join(
        os.path.dirname(workspace_root),
        "rules-vs-examples-dataset",
        "data",
    )
    if os.path.isdir(sibling_data):
        return sibling_data
    return local_data


def unwrap_item_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert paper-facing JSONL rows into task-runner item records."""
    item = record.get("item")
    if not isinstance(item, dict):
        return record

    out = dict(item)
    for key in ("id", "difficulty", "split", "task"):
        if key not in out and key in record:
            out[key] = record[key]
    if "source_row" not in out:
        out["source_row"] = {
            key: record[key]
            for key in ("id", "task", "difficulty", "split", "label", "condition", "source")
            if key in record
        }
    return out


def load_records(path: str, *, unwrap_items: bool = True) -> List[Dict[str, Any]]:
    """Load list-like JSON or JSONL records."""
    if path.endswith(".jsonl"):
        rows: List[Dict[str, Any]] = []
        with open(path, "r") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        with open(path, "r") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            rows = list(data["data"])
        else:
            raise ValueError(f"Expected a list, JSONL records, or a dict with a data list in {path}")

    if unwrap_items:
        return [unwrap_item_record(row) if isinstance(row, dict) else row for row in rows]
    return rows


def select_latest_file(patterns: Sequence[str]) -> Optional[str]:
    """Return the latest matching file, preferring numeric suffixes when present."""
    matches: List[str] = []
    for pattern in patterns:
        matches.extend(glob(pattern))
    if not matches:
        return None

    def score(path: str) -> tuple[int, str]:
        base = os.path.basename(path)
        number = -1
        stem = os.path.splitext(base)[0]
        for token in reversed(stem.replace("-", "_").split("_")):
            if token.startswith("n") and token[1:].isdigit():
                number = int(token[1:])
                break
            if token.isdigit():
                number = int(token)
                break
        return number, path

    return max(matches, key=score)
