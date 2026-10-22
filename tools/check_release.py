#!/usr/bin/env python3
"""Require public version metadata and the latest changelog entry to agree."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
CHANGELOG_RE = re.compile(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\] - [0-9]{4}-[0-9]{2}-[0-9]{2}$", re.MULTILINE)
CITATION_RE = re.compile(r"^version: ([0-9]+\.[0-9]+\.[0-9]+)$", re.MULTILINE)
README_RE = re.compile(r"Version \[`([0-9]+\.[0-9]+\.[0-9]+)`\]\(https://github\.com/aownown/ego2robot-repro/releases/tag/v\1\)")


def one(pattern: re.Pattern[str], text: str, label: str) -> str:
    matches = pattern.findall(text)
    if len(matches) != 1:
        raise ValueError(f"{label}: expected exactly one version, found {matches}")
    version = matches[0]
    if not VERSION_RE.fullmatch(version):
        raise ValueError(f"{label}: invalid semantic version: {version}")
    return version


def main() -> int:
    try:
        citation = one(CITATION_RE, (ROOT / "CITATION.cff").read_text(), "CITATION.cff")
        readme = one(README_RE, (ROOT / "README.md").read_text(), "README.md")
        changelog_versions = CHANGELOG_RE.findall((ROOT / "CHANGELOG.md").read_text())
        if not changelog_versions:
            raise ValueError("CHANGELOG.md: no dated release entry found")
        changelog = changelog_versions[0]
        versions = {"CITATION.cff": citation, "README.md": readme, "CHANGELOG.md": changelog}
        if len(set(versions.values())) != 1:
            details = ", ".join(f"{name}={version}" for name, version in versions.items())
            raise ValueError(f"release versions disagree: {details}")
        if not (ROOT / "docs/RELEASING.md").is_file():
            raise ValueError("missing release checklist: docs/RELEASING.md")
        if not (ROOT / ".github/dependabot.yml").is_file():
            raise ValueError("missing dependency update configuration: .github/dependabot.yml")
    except (OSError, ValueError) as error:
        print(f"Release consistency check failed: {error}")
        return 1
    print(f"Release consistency passed (version {citation})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
