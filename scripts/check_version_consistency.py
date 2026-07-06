"""Assert that Modeling_Tool.__version__ / setup.py / pyproject.toml agree.

Ships as a standalone script so both the version-consistency CI workflow
and a developer's pre-commit / manual invocation can call the same
implementation. Zero third-party deps: only stdlib.

Run from repo root:
    python scripts/check_version_consistency.py

Exit codes:
    0 — all three sources agree
    1 — mismatch (message names the values found and the file paths)
    2 — a source file is missing or malformed
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

INIT_PATH = REPO_ROOT / "Modeling_Tool" / "__init__.py"
SETUP_PATH = REPO_ROOT / "setup.py"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


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

    We assert on the *literal fallback default*, not on os.environ at check
    time — the intent is that the pinned string in the repo agrees with
    the other sources of truth, regardless of what SMF_VERSION happens to
    be in the CI shell.
    """
    text = path.read_text(encoding="utf-8")
    # First try the environ-fallback pattern.
    match = re.search(
        r'os\.environ\.get\(\s*["\']SMF_VERSION["\']\s*,\s*["\'](?P<v>[^"\']+)["\']',
        text,
    )
    if match is not None:
        return match.group("v")
    # Fall back to a plain `version="X.Y.Z"` kwarg on setup(...).
    match = re.search(r'version\s*=\s*["\'](?P<v>[^"\']+)["\']', text)
    if match is None:
        raise ValueError(f"version kwarg not found in {path}")
    return match.group("v")


def read_pyproject_version(path: Path) -> str:
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    try:
        return data["project"]["version"]
    except KeyError as exc:
        raise ValueError(f"[project].version missing in {path}") from exc


def main() -> int:
    try:
        init_v = read_init_version(INIT_PATH)
        setup_v = read_setup_version(SETUP_PATH)
        pyproj_v = read_pyproject_version(PYPROJECT_PATH)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    versions = {
        str(INIT_PATH.relative_to(REPO_ROOT)): init_v,
        str(SETUP_PATH.relative_to(REPO_ROOT)): setup_v,
        str(PYPROJECT_PATH.relative_to(REPO_ROOT)): pyproj_v,
    }

    unique = set(versions.values())
    if len(unique) == 1:
        version = unique.pop()
        print(f"OK — all three sources agree on version {version}")
        for path, ver in versions.items():
            print(f"    {path}: {ver}")
        return 0

    print("FAIL — version drift detected across sources:", file=sys.stderr)
    for path, ver in versions.items():
        print(f"    {path}: {ver}", file=sys.stderr)
    print(
        "\nAll three of Modeling_Tool/__init__.py, setup.py, and pyproject.toml "
        "must agree before release. Fix the mismatch and re-run.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
