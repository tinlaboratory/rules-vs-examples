"""Utilities for loading prompt templates from Markdown files."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
RULES_MODE_PROMPT_DIR = REPO_ROOT / "rules-mode-prompts"
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _normalize_heading(text: str) -> str:
    return " ".join(text.strip().lower().split())


@lru_cache(maxsize=None)
def _read_markdown(filename: str) -> list[str]:
    path = RULES_MODE_PROMPT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Rules-mode prompt Markdown file not found: {path}")
    return path.read_text(encoding="utf-8").splitlines()


@lru_cache(maxsize=None)
def get_markdown_section(filename: str, heading_path: tuple[str, ...]) -> str:
    """Return the content under a nested Markdown heading path.

    The path is matched as a suffix, so files may have a top-level title before
    the task/mode headings.
    """
    target = tuple(_normalize_heading(part) for part in heading_path)
    lines = _read_markdown(filename)
    stack: list[tuple[int, str]] = []
    start: int | None = None
    start_level: int | None = None

    for idx, line in enumerate(lines):
        match = _HEADING_RE.match(line)
        if not match:
            continue

        level = len(match.group(1))
        title = _normalize_heading(match.group(2))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        if start is not None and start_level is not None and level <= start_level:
            return "\n".join(lines[start:idx]).strip("\n")

        stack_titles = tuple(item[1] for item in stack)
        if len(stack_titles) >= len(target) and stack_titles[-len(target) :] == target:
            start = idx + 1
            start_level = level

    if start is None:
        display_path = " > ".join(heading_path)
        raise KeyError(f"Missing Markdown prompt section {display_path!r} in {filename}")
    return "\n".join(lines[start:]).strip("\n")


def render_markdown_prompt(filename: str, heading_path: Iterable[str], **values: object) -> str:
    template = get_markdown_section(filename, tuple(heading_path))
    return template.format(**values)
