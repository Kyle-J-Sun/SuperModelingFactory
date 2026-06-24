#!/usr/bin/env python3
# =============================================================================
# verify_protection.py
# -----------------------------------------------------------------------------
# Guard-rail script that prevents accidental leaks of closed-source modules.
#
# Runs three sanity checks:
#
#   [check 1]  setup.py CLOSED_SOURCE_MODULES  ==  gen_stubs.py MODULES
#   [check 2]  Every closed-source .py has a sibling .pyi  AND  every closed
#              -source .py is referenced by an `exclude` line in MANIFEST.in
#   [check 3]  If --wheel <path> is supplied (or wheelhouse/*.whl auto-found),
#              verify the wheel contains NO closed-source .py and DOES contain
#              one .so/.pyd per closed-source module.
#
# Exit codes:
#   0  -- all checks passed
#   1  -- at least one check failed (suitable for CI fail-fast)
#
# Usage:
#   python scripts/verify_protection.py                # checks 1 & 2
#   python scripts/verify_protection.py --wheel a.whl  # also check 3
#   python scripts/verify_protection.py --wheelhouse   # scan wheelhouse/ dir
#
# Designed to be invoked from .github/workflows/build.yml so a PR that forgets
# to update one of the three config locations gets blocked before merge.
# =============================================================================
from __future__ import annotations

import argparse
import ast
import re
import sys
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Source-of-truth extractors
# -----------------------------------------------------------------------------
def _extract_module_list(py_file: Path, var_name: str) -> list[tuple[str, str]]:
    """Pull the `[(dotted, src), ...]` literal assigned to *var_name*.

    Uses AST so we don't need to import setup.py / gen_stubs.py (which would
    drag in Cython etc. just to read a constant).
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = []
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
            value = node.value
        elif isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            value = node.value
        if var_name in targets and isinstance(value, (ast.List, ast.Tuple)):
            out: list[tuple[str, str]] = []
            for elt in value.elts:
                if isinstance(elt, ast.Tuple) and len(elt.elts) == 2:
                    a, b = elt.elts
                    if isinstance(a, ast.Constant) and isinstance(b, ast.Constant):
                        out.append((a.value, b.value))
            return out
    raise SystemExit(f"[verify] could not find {var_name} in {py_file}")


def _extract_manifest_excludes(manifest: Path) -> set[str]:
    """Return the set of paths after every `exclude ` directive."""
    out: set[str] = set()
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        # Match `exclude PATH` but NOT `recursive-exclude` or `global-exclude`
        m = re.match(r"^exclude\s+(\S+)\s*$", line)
        if m:
            out.add(m.group(1))
    return out


# -----------------------------------------------------------------------------
# Pretty reporting helpers
# -----------------------------------------------------------------------------
RED = "\033[31m"; GREEN = "\033[32m"; YELLOW = "\033[33m"; RESET = "\033[0m"

# Windows' default cp1252 / cp936 stdout can't encode the Unicode tick / cross
# glyphs, which makes the script crash on a successful check. Fall back to
# plain ASCII markers if stdout can't handle them.
try:
    "✔✘".encode(sys.stdout.encoding or "utf-8")
    _TICK, _CROSS = "✔", "✘"
except (UnicodeEncodeError, LookupError):
    _TICK, _CROSS = "OK", "FAIL"


def _ok(msg: str) -> None:
    print(f"{GREEN}{_TICK}{RESET} {msg}")


def _fail(msg: str, items: Iterable[str] = ()) -> None:
    print(f"{RED}{_CROSS} {msg}{RESET}")
    for it in items:
        print(f"    - {it}")


def _warn(msg: str) -> None:
    print(f"{YELLOW}!{RESET} {msg}")


# -----------------------------------------------------------------------------
# Individual checks
# -----------------------------------------------------------------------------
def check_setup_stub_parity() -> bool:
    """setup.py CLOSED_SOURCE_MODULES must equal gen_stubs.py MODULES."""
    setup_list  = _extract_module_list(ROOT / "setup.py",      "CLOSED_SOURCE_MODULES")
    stubs_list  = _extract_module_list(ROOT / "gen_stubs.py",  "MODULES")

    a = set(setup_list)
    b = set(stubs_list)
    only_setup = a - b
    only_stubs = b - a

    if not only_setup and not only_stubs:
        _ok(f"setup.py and gen_stubs.py agree on {len(setup_list)} closed-source modules")
        return True

    if only_setup:
        _fail("In setup.py but missing from gen_stubs.py:", [f"{m}  ({s})" for m, s in sorted(only_setup)])
    if only_stubs:
        _fail("In gen_stubs.py but missing from setup.py:", [f"{m}  ({s})" for m, s in sorted(only_stubs)])
    return False


def check_pyi_and_manifest() -> bool:
    """Every closed-source .py must have a .pyi sibling AND an exclude line."""
    modules = _extract_module_list(ROOT / "setup.py", "CLOSED_SOURCE_MODULES")
    manifest_excludes = _extract_manifest_excludes(ROOT / "MANIFEST.in")

    missing_pyi: list[str] = []
    missing_excl: list[str] = []
    missing_py:  list[str] = []

    for _dotted, src in modules:
        src_path = ROOT / src
        if not src_path.exists():
            missing_py.append(src)
            continue
        pyi_path = src_path.with_suffix(".pyi")
        if not pyi_path.exists():
            missing_pyi.append(str(pyi_path.relative_to(ROOT)))
        # MANIFEST exclude lines are repository-relative POSIX paths
        if src not in manifest_excludes:
            missing_excl.append(src)

    passed = not (missing_pyi or missing_excl or missing_py)
    if passed:
        _ok(f"All {len(modules)} modules have .py + .pyi + MANIFEST.in exclude entries")
        return True

    if missing_py:
        _fail("Listed in setup.py but .py file does not exist on disk:", missing_py)
    if missing_pyi:
        _fail(".pyi stub missing (run `python gen_stubs.py` to regenerate):", missing_pyi)
    if missing_excl:
        _fail("Closed-source .py NOT excluded in MANIFEST.in -- sdist would leak source:", missing_excl)
    return False


def check_wheel(wheel_path: Path) -> bool:
    """Open the wheel, ensure no closed-source .py leaked and all .so present."""
    modules = _extract_module_list(ROOT / "setup.py", "CLOSED_SOURCE_MODULES")
    closed_py_basenames = {Path(src).name for _, src in modules}
    closed_dotted = {dotted for dotted, _ in modules}

    with zipfile.ZipFile(wheel_path) as zf:
        names = zf.namelist()

    # ---- 1. no closed-source .py allowed ----
    leaked = [n for n in names
              if n.endswith(".py") and Path(n).name in closed_py_basenames]

    # ---- 2. at least one compiled artefact per closed-source module ----
    # Naming pattern: "<dotted>.<cpython-...>.so" or "<dotted>.pyd".
    # Wheels organise files by package path, e.g.
    #   Modeling_Tool/WOE/WOE_Master.cpython-312-x86_64-linux-gnu.so
    def _has_compiled(dotted: str) -> bool:
        pkg_path = dotted.replace(".", "/")
        stem = Path(pkg_path).name
        parent = str(Path(pkg_path).parent)
        for n in names:
            np = Path(n)
            if str(np.parent) != parent:
                continue
            if np.name.startswith(stem + ".") and (np.suffix in (".so", ".pyd")):
                return True
        return False

    missing_compiled = [d for d in closed_dotted if not _has_compiled(d)]

    passed = not (leaked or missing_compiled)
    if passed:
        _ok(f"Wheel {wheel_path.name}: 0 leaked .py, all {len(closed_dotted)} extensions present")
        return True

    if leaked:
        _fail(f"Wheel {wheel_path.name} LEAKS closed-source .py:", leaked)
    if missing_compiled:
        _fail(f"Wheel {wheel_path.name} missing compiled extensions:", sorted(missing_compiled))
    return False


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wheel", type=Path, action="append", default=[],
                    help="Inspect a built wheel (may be repeated).")
    ap.add_argument("--wheelhouse", action="store_true",
                    help="Auto-scan ./wheelhouse/*.whl and ./dist/*.whl")
    args = ap.parse_args()

    print("=" * 70)
    print(" SuperModelingFactory -- protection verifier")
    print("=" * 70)

    ok = True
    ok &= check_setup_stub_parity()
    ok &= check_pyi_and_manifest()

    wheels: list[Path] = list(args.wheel)
    if args.wheelhouse:
        for d in ("wheelhouse", "dist"):
            wheels.extend(sorted((ROOT / d).glob("*.whl")))

    if wheels:
        for w in wheels:
            if not w.exists():
                _fail(f"Wheel not found: {w}")
                ok = False
                continue
            ok &= check_wheel(w)
    else:
        _warn("No wheel supplied -- skipping wheel-content check.")
        _warn("Run with `--wheelhouse` or `--wheel path.whl` for full verification.")

    print()
    if ok:
        print(f"{GREEN}ALL CHECKS PASSED.{RESET}")
        return 0
    print(f"{RED}VERIFICATION FAILED -- fix the issues above before merging / releasing.{RESET}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
