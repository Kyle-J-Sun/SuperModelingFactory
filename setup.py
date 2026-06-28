# =============================================================================
# SuperModelingFactory — Cython build configuration
# -----------------------------------------------------------------------------
# Compiles the proprietary core modules (WOE / Feature / Model / Eval / Sample /
# selected Core algorithms) into platform-specific .so / .pyd extensions so that
# the original Python source is NOT shipped in the wheel.
#
# The remaining modules (Core data plumbing, UAT, ExcelMaster, Report) stay as
# plain .py and are distributed as a regular open-source Python package.
#
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# Licensed under the Business Source License 1.1 (see LICENSE).
# =============================================================================
from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Cython is required to build SuperModelingFactory.\n"
        "Install it first:  pip install 'cython>=3.0'"
    ) from exc


ROOT = Path(__file__).parent


# -----------------------------------------------------------------------------
# Proprietary modules to be compiled into .so / .pyd (source NOT shipped).
# Keep this list in sync with the closed-source decision matrix in README.
# -----------------------------------------------------------------------------
CLOSED_SOURCE_MODULES: list[tuple[str, str]] = [
    # ---- WOE ----
    ("Modeling_Tool.WOE.WOE_Master",          "Modeling_Tool/WOE/WOE_Master.py"),
    ("Modeling_Tool.WOE.WOE_Monotone_Binner", "Modeling_Tool/WOE/WOE_Monotone_Binner.py"),
    ("Modeling_Tool.WOE.WOE_Plot_Tool",       "Modeling_Tool/WOE/WOE_Plot_Tool.py"),
    ("Modeling_Tool.WOE.WOE_Report_Builder",  "Modeling_Tool/WOE/WOE_Report_Builder.py"),
    ("Modeling_Tool.WOE.WOE_Tool",            "Modeling_Tool/WOE/WOE_Tool.py"),
    ("Modeling_Tool.WOE.plot_woe_tool",       "Modeling_Tool/WOE/plot_woe_tool.py"),
    # ---- Feature ----
    ("Modeling_Tool.Feature.Distribution_Tool", "Modeling_Tool/Feature/Distribution_Tool.py"),
    ("Modeling_Tool.Feature.Feature_Insights",  "Modeling_Tool/Feature/Feature_Insights.py"),
    ("Modeling_Tool.Feature.PSI_Tool",          "Modeling_Tool/Feature/PSI_Tool.py"),
    # ---- Model ----
    ("Modeling_Tool.Model.Backward_Tool", "Modeling_Tool/Model/Backward_Tool.py"),
    ("Modeling_Tool.Model.GBM_Tool",      "Modeling_Tool/Model/GBM_Tool.py"),
    ("Modeling_Tool.Model.LRM_Tool",      "Modeling_Tool/Model/LRM_Tool.py"),
    # ---- Eval ----
    ("Modeling_Tool.Eval.Evaluation_Tool", "Modeling_Tool/Eval/Evaluation_Tool.py"),
    ("Modeling_Tool.Eval.Model_Eval_Tool", "Modeling_Tool/Eval/Model_Eval_Tool.py"),
    ("Modeling_Tool.Eval.evaluate_model",  "Modeling_Tool/Eval/evaluate_model.py"),
    # ---- Sample ----
    ("Modeling_Tool.Sample.Distribution_Adaptation", "Modeling_Tool/Sample/Distribution_Adaptation.py"),
    ("Modeling_Tool.Sample.Reject_Infer",            "Modeling_Tool/Sample/Reject_Infer.py"),
    ("Modeling_Tool.Sample.Sample_Split",            "Modeling_Tool/Sample/Sample_Split.py"),
    # ---- Core (algorithmic subset) ----
    ("Modeling_Tool.Core.Binning_Tool",   "Modeling_Tool/Core/Binning_Tool.py"),
    ("Modeling_Tool.Core.kDataFrame",     "Modeling_Tool/Core/kDataFrame.py"),
    ("Modeling_Tool.Core.XOR_Encryptor",  "Modeling_Tool/Core/XOR_Encryptor.py"),
    ("Modeling_Tool.Core.Slope_Tool",     "Modeling_Tool/Core/Slope_Tool.py"),
]


extensions = [
    Extension(
        name=mod,
        sources=[src],
        # Compile-time defines kept minimal; numpy headers are optional and only
        # needed if individual modules use the legacy C-API.
    )
    for mod, src in CLOSED_SOURCE_MODULES
]

# Fully-qualified module names of every closed-source target (dotted form).
# Used by the custom build_py command below to strip the original .py source
# from the wheel — only the compiled .so / .pyd extension ships.
CLOSED_SOURCE_DOTTED = {mod for mod, _ in CLOSED_SOURCE_MODULES}


class build_py_strip_closed_source(_build_py):
    """Custom build_py that excludes closed-source .py files from the wheel.

    setuptools' default find_packages() + build_py would happily ship the
    original Python source alongside the compiled extension, defeating the
    purpose of the protection layer. We filter the (package, module, file)
    tuples returned by `find_package_modules` so the .py source is never
    copied into the build tree — only the Cython-generated .so / .pyd is
    packaged.
    """

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        filtered = []
        for pkg, mod, filepath in modules:
            dotted = f"{pkg}.{mod}"
            if dotted in CLOSED_SOURCE_DOTTED:
                # Skip — only the compiled extension and the .pyi stub ship.
                continue
            if mod.startswith("test_") or mod == "conftest":
                # Tests stay in the repo for `make test`, but never ship.
                continue
            filtered.append((pkg, mod, filepath))
        return filtered


# -----------------------------------------------------------------------------
# Cython directives
#   language_level   : Python 3 syntax
#   embedsignature   : OFF — do not embed Python signatures in the .so binary
#                      (lowers reverse-engineering surface; users still get
#                      signatures via the shipped .pyi stubs)
#   binding          : True — produce CPython-compatible function objects
#   boundscheck/wraparound  : keep ON (safe defaults). Flip OFF per-module
#                              via in-file `# cython: ...` if a hot loop needs it.
# -----------------------------------------------------------------------------
CYTHON_DIRECTIVES = {
    "language_level": "3",
    "embedsignature": False,
    "binding": True,
    "boundscheck": True,
    "wraparound": True,
}


def _read(p: str) -> str:
    f = ROOT / p
    return f.read_text(encoding="utf-8") if f.exists() else ""


setup(
    name="SuperModelingFactory",
    version=os.environ.get("SMF_VERSION", "0.1.3"),
    description="Credit risk modeling factory: WOE binning, scorecards, LightGBM, Excel reporting.",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    author="Kyle Sun",
    author_email="jingkai.sun20@alumni.imperial.ac.uk",
    url="https://github.com/Kyle-J-Sun/SuperModelingFactory",
    license="BUSL-1.1",
    python_requires=">=3.10",
    # find_packages picks up every namespace that contains an __init__.py
    # — Modeling_Tool, Modeling_Tool.Core, Modeling_Tool.WOE, ExcelMaster, Report, etc.
    packages=find_packages(
        exclude=[
            "*.backup", "*.backup.*", "backup.*", "backup",
            "tests", "tests.*",
        ],
    ),
    # Ship the .pyi stubs alongside the compiled extensions so IDEs keep type
    # hints and the FINGERPRINT/copyright markers remain on disk.
    package_data={
        "Modeling_Tool.WOE":     ["*.pyi"],
        "Modeling_Tool.Feature": ["*.pyi"],
        "Modeling_Tool.Model":   ["*.pyi"],
        "Modeling_Tool.Eval":    ["*.pyi"],
        "Modeling_Tool.Sample":  ["*.pyi"],
        "Modeling_Tool.Core":    ["*.pyi"],
    },
    ext_modules=cythonize(
        extensions,
        compiler_directives=CYTHON_DIRECTIVES,
        # We want a deterministic build; do not regenerate .c on every run.
        force=True,
        # Quiet mode keeps CI logs readable; flip to False for debugging.
        quiet=False,
    ),
    cmdclass={"build_py": build_py_strip_closed_source},
    zip_safe=False,
    # include_package_data=False (default) prevents Cython-generated .c
    # intermediates from leaking into the wheel; only .pyi stubs declared in
    # package_data above are shipped alongside the compiled extensions.
    include_package_data=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: Other/Proprietary License",
    ],
)
