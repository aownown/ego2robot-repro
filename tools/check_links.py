#!/usr/bin/env python3
"""Check local targets in Markdown links and images without network access."""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SCHEMES = ("http://", "https://", "mailto:", "data:")
SKIP_PARTS = {".git", ".venv", "envs", "repos", "__pycache__"}


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not SKIP_PARTS.intersection(path.parts)
    )


def local_target(raw: str) -> tuple[str, str]:
    target = raw.strip().split(maxsplit=1)[0].strip("<>")
    if not target or target.startswith("#") or target.startswith(SCHEMES):
        return "", ""
    path, separator, anchor = target.partition("#")
    return unquote(path), separator + anchor if separator else ""


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text(errors="replace")
    for match in LINK_RE.finditer(text):
        target, anchor = local_target(match.group(1))
        if not target:
            continue
        destination = (path.parent / target).resolve()
        try:
            destination.relative_to(ROOT)
        except ValueError:
            errors.append(
                f"{path.relative_to(ROOT)}: local link escapes repository: {match.group(1)}"
            )
            continue
        if not destination.exists():
            errors.append(
                f"{path.relative_to(ROOT)}: missing local target: {match.group(1)}"
            )
        elif anchor and destination.is_file() and destination.suffix.lower() != ".md":
            errors.append(
                f"{path.relative_to(ROOT)}: anchor on non-Markdown target: {match.group(1)}"
            )
    return errors


def main() -> int:
    errors = [error for path in markdown_files() for error in validate(path)]
    if errors:
        print("Markdown link check failed:")
        for error in errors:
            print(f" - {error}")
        return 1
    print(f"Markdown links passed ({len(markdown_files())} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
