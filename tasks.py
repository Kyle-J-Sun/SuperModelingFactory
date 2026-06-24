# =============================================================================
# tasks.py \u2014 cross-platform alternative to the Makefile
# -----------------------------------------------------------------------------
# Windows users can run e.g.  `python tasks.py build`  instead of `make build`.
# Linux / macOS users may still prefer the Makefile.
#
# All commands are pure-Python; no `invoke` / `nox` dependency required.
# =============================================================================
from __future__ import annotations

import argparse
import os
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
    """Editable install with Cython available."""
    _run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    _run([PY, "-m", "pip", "install", "cython>=3.0", "build"])
    _run([PY, "-m", "pip", "install", "-e", "."])


def stub(_args) -> None:
    """Regenerate every .pyi stub."""
    _run([PY, "gen_stubs.py"])


def compile_all(_args) -> None:
    """Build all .so / .pyd in place."""
    _run([PY, "setup.py", "build_ext", "--inplace"])


def build(_args) -> None:
    """Build a wheel for the current platform."""
    _run([PY, "-m", "build", "--wheel"])


def sdist(_args) -> None:
    _run([PY, "-m", "build", "--sdist"])


def verify(_args) -> None:
    _run([PY, "scripts/verify_protection.py", "--wheelhouse"])


def test(_args) -> None:
    _run([PY, "-c",
          "import Modeling_Tool, Modeling_Tool.WOE.WOE_Master, "
          "Modeling_Tool.Model.LRM_Tool, Modeling_Tool.Feature.PSI_Tool; "
          "print('imports OK')"])


def info(_args) -> None:
    """List closed-source modules + fingerprints."""
    sys.path.insert(0, str(ROOT))
    import gen_stubs  # type: ignore
    for dotted, src in gen_stubs.MODULES:
        print(f"{dotted:55s}  {gen_stubs.fingerprint(dotted)}  ({src})")


def clean(_args) -> None:
    for pat in ("*.c", "*.so", "*.pyd", "*.html"):
        for p in (ROOT / "Modeling_Tool").rglob(pat):
            p.unlink()
            print("rm", p.relative_to(ROOT))
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

    pyproject = ROOT / "pyproject.toml"
    new_text = re.sub(
        r"^version\s*=.*", f'version = "{version}"',
        pyproject.read_text(encoding="utf-8"),
        count=1, flags=re.M,
    )
    pyproject.write_text(new_text, encoding="utf-8")
    print(f"[release] version bumped to {version}")

    verify(None)

    _run(["git", "add", "pyproject.toml"])
    _run(["git", "commit", "-m", f"chore: bump version to {version}"])
    _run(["git", "tag", "-a", f"v{version}", "-m", f"Release v{version}"])

    print()
    print("Ready to push. Run:")
    print("    git push origin main")
    print(f"    git push origin v{version}")
    print()
    print("GitHub Actions will then build wheels and create the Release.")


# --- CLI ---------------------------------------------------------------------

TASKS = {
    "install": install,
    "stub":    stub,
    "compile": compile_all,
    "build":   build,
    "sdist":   sdist,
    "verify":  verify,
    "test":    test,
    "info":    info,
    "clean":   clean,
    "release": release,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="SMF developer tasks")
    sub = ap.add_subparsers(dest="task", required=True)

    for name in TASKS:
        sp = sub.add_parser(name, help=TASKS[name].__doc__ or "")
        if name == "release":
            sp.add_argument("version", help="Semver string, e.g. 0.1.1")

    args = ap.parse_args()
    TASKS[args.task](args)


if __name__ == "__main__":
    main()
