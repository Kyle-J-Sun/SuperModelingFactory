# =============================================================================
# tasks.py — cross-platform alternative to the Makefile
# -----------------------------------------------------------------------------
# Windows users can run e.g.  `python tasks.py build`  instead of `make build`.
# Linux / macOS users may still prefer the Makefile.
#
# All commands are pure-Python; no `invoke` / `nox` dependency required.
# =============================================================================
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(cmd: list[str], **kw) -> None:
    print(">>>", " ".join(cmd))
    subprocess.check_call(cmd, cwd=ROOT, **kw)


# --- task implementations ----------------------------------------------------

def install(_args) -> None:
    """Editable install."""
    _run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    _run([PY, "-m", "pip", "install", "-e", "."])


def build(_args) -> None:
    """Build a wheel for the current platform."""
    _run([PY, "-m", "pip", "install", "--upgrade", "build"])
    _run([PY, "-m", "build", "--wheel"])


def sdist(_args) -> None:
    _run([PY, "-m", "pip", "install", "--upgrade", "build"])
    _run([PY, "-m", "build", "--sdist"])


def verify(_args) -> None:
    _run([PY, "-m", "compileall", "-q", "Modeling_Tool", "ExcelMaster", "Report"])
    _run([PY, "-m", "pip", "install", "--upgrade", "build"])
    _run([PY, "-m", "build", "--sdist", "--wheel"])


def test(_args) -> None:
    _run([PY, "-c",
          "import Modeling_Tool, Modeling_Tool.WOE.WOE_Master, "
          "Modeling_Tool.Model.LRM_Tool, Modeling_Tool.Feature.PSI_Tool, "
          "Modeling_Tool.Pipeline; print('imports OK')"])


def clean(_args) -> None:
    for p in ROOT.rglob("__pycache__"):
        shutil.rmtree(p, ignore_errors=True)
    for d in ("build", "dist", "wheelhouse"):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    for egg in ROOT.glob("*.egg-info"):
        shutil.rmtree(egg, ignore_errors=True)


def release(args) -> None:
    """End-to-end release: bump version -> verify -> commit -> tag.

    Does NOT push automatically (caller must run `git push` after reviewing).
    """
    version = args.version
    if not re.match(r"^\d+\.\d+\.\d+([abrc]\d+|\.post\d+)?$", version):
        sys.exit(f"[release] invalid version string: {version!r}")

    bumps = [
        (ROOT / "pyproject.toml", r"^version\s*=.*", f'version = "{version}"'),
        (ROOT / "setup.py", r'SMF_VERSION", "[^"]+"', f'SMF_VERSION", "{version}"'),
        (ROOT / "Modeling_Tool" / "__init__.py", r'__version__ = "[^"]+"', f'__version__ = "{version}"'),
        (ROOT / "README.md", r"^-\s+\*\*Version\*\*:\s*\S+", f"- **Version**: {version}"),
        (ROOT / "Modeling_Tool" / "README.md", r"^-\s+\*\*Version\*\*:\s*\S+", f"- **Version**: {version}"),
    ]
    for path, pattern, repl in bumps:
        text = path.read_text(encoding="utf-8")
        path.write_text(re.sub(pattern, repl, text, count=1, flags=re.M), encoding="utf-8")
    print(f"[release] version bumped to {version}")

    verify(None)

    _run([
        "git",
        "add",
        "pyproject.toml",
        "setup.py",
        "Modeling_Tool/__init__.py",
        "README.md",
        "Modeling_Tool/README.md",
    ])
    _run(["git", "commit", "-m", f"chore: bump version to {version}"])
    _run(["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"])

    print()
    print("Ready to push. Run:")
    print("    git push origin main")
    print(f"    git push origin v{version}")
    print()
    print("Also sync SuperModelingFactory_doc / SuperModelingFactory_pytest version references.")
    print("GitHub Actions will then build wheels and create the Release.")


# --- CLI ---------------------------------------------------------------------

TASKS = {
    "install": install,
    "build":   build,
    "sdist":   sdist,
    "verify":  verify,
    "test":    test,
    "clean":   clean,
    "release": release,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="SMF developer tasks")
    sub = ap.add_subparsers(dest="task", required=True)

    for name in TASKS:
        sp = sub.add_parser(name, help=TASKS[name].__doc__ or "")
        if name == "release":
            sp.add_argument("version", help="Semver string, e.g. 0.3.6")

    args = ap.parse_args()
    TASKS[args.task](args)


if __name__ == "__main__":
    main()
