"""Assert that all checked version references agree.

Ships as a standalone script so both the version-consistency CI workflow
and a developer's pre-commit / manual invocation can call the same
implementation. Zero third-party deps: only stdlib.

Run from repo root:
    python scripts/check_version_consistency.py

Exit codes:
    0 - all version sources agree
    1 - mismatch (message names the values found and the file paths)
    2 - a source file is missing or malformed
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11 local developer environments.
    tomllib = None

REPO_ROOT = Path(__file__).resolve().parent.parent

INIT_PATH = REPO_ROOT / "Modeling_Tool" / "__init__.py"
SETUP_PATH = REPO_ROOT / "setup.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
README_PATH = REPO_ROOT / "README.md"
MODELING_README_PATH = REPO_ROOT / "Modeling_Tool" / "README.md"


def read_init_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*["\'](?P<v>[^"\']+)["\']', text, re.MULTILINE)
    if match is None:
        raise ValueError(f"__version__ assignment not found in {path}")
    return match.group("v")


def read_setup_version(path: Path) -> str:
    """Extract the fallback default from setup.py.

    The current setup.py uses:
        version=os.environ.get("SMF_VERSION", "X.Y.Z")

    We assert on the literal fallback default, not on os.environ at check
    time. The intent is that the pinned string in the repo agrees with the
    other sources of truth, regardless of what SMF_VERSION happens to be in
    the CI shell.
    """
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r'os\.environ\.get\(\s*["\']SMF_VERSION["\']\s*,\s*["\'](?P<v>[^"\']+)["\']',
        text,
    )
    if match is not None:
        return match.group("v")

    match = re.search(r'version\s*=\s*["\'](?P<v>[^"\']+)["\']', text)
    if match is None:
        raise ValueError(f"version kwarg not found in {path}")
    return match.group("v")


def read_pyproject_version(path: Path) -> str:
    if tomllib is None:
        text = path.read_text(encoding="utf-8")
        match = re.search(
            r'(?ms)^\[project\].*?^version\s*=\s*["\'](?P<v>[^"\']+)["\']',
            text,
        )
        if match is None:
            raise ValueError(f"[project].version missing in {path}")
        return match.group("v")

    with path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data["project"]["version"]
    except KeyError as exc:
        raise ValueError(f"[project].version missing in {path}") from exc


def read_readme_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^\s*-\s+\*\*Version\*\*:\s*(?P<v>\S+)\s*$", text, re.MULTILINE)
    if match is None:
        raise ValueError(f"README version footer not found in {path}")
    return match.group("v")


def main() -> int:
    try:
        versions = {
            str(INIT_PATH.relative_to(REPO_ROOT)): read_init_version(INIT_PATH),
            str(SETUP_PATH.relative_to(REPO_ROOT)): read_setup_version(SETUP_PATH),
            str(PYPROJECT_PATH.relative_to(REPO_ROOT)): read_pyproject_version(PYPROJECT_PATH),
            str(README_PATH.relative_to(REPO_ROOT)): read_readme_version(README_PATH),
            str(MODELING_README_PATH.relative_to(REPO_ROOT)): read_readme_version(MODELING_README_PATH),
        }
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    unique = set(versions.values())
    if len(unique) == 1:
        version = unique.pop()
        print(f"OK - all version sources agree on version {version}")
        for path, ver in versions.items():
            print(f"    {path}: {ver}")
        return 0

    print("FAIL - version drift detected across sources:", file=sys.stderr)
    for path, ver in versions.items():
        print(f"    {path}: {ver}", file=sys.stderr)
    print(
        "\nAll version sources must agree before release. Fix the mismatch and re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
