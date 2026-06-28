# =============================================================================
# SuperModelingFactory — Cython build configuration
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
    raise SystemExit("Cython is required to build SuperModelingFactory.\nInstall it first:  pip install 'cython>=3.0'") from exc

ROOT = Path(__file__).parent

CLOSED_SOURCE_MODULES: list[tuple[str, str]] = [
    ("Modeling_Tool.weighted_integration", "Modeling_Tool/weighted_integration.py"),
    ("Modeling_Tool.WOE.WOE_Master",          "Modeling_Tool/WOE/WOE_Master.py"),
    ("Modeling_Tool.WOE.WOE_Monotone_Binner", "Modeling_Tool/WOE/WOE_Monotone_Binner.py"),
    ("Modeling_Tool.WOE.WOE_Plot_Tool",       "Modeling_Tool/WOE/WOE_Plot_Tool.py"),
    ("Modeling_Tool.WOE.WOE_Report_Builder",  "Modeling_Tool/WOE/WOE_Report_Builder.py"),
    ("Modeling_Tool.WOE.WOE_Tool",            "Modeling_Tool/WOE/WOE_Tool.py"),
    ("Modeling_Tool.WOE.plot_woe_tool",       "Modeling_Tool/WOE/plot_woe_tool.py"),
    ("Modeling_Tool.Feature.Distribution_Tool", "Modeling_Tool/Feature/Distribution_Tool.py"),
    ("Modeling_Tool.Feature.Feature_Insights",  "Modeling_Tool/Feature/Feature_Insights.py"),
    ("Modeling_Tool.Feature.PSI_Tool",          "Modeling_Tool/Feature/PSI_Tool.py"),
    ("Modeling_Tool.Model.Backward_Tool",    "Modeling_Tool/Model/Backward_Tool.py"),
    ("Modeling_Tool.Model.GBM_Tool",         "Modeling_Tool/Model/GBM_Tool.py"),
    ("Modeling_Tool.Model.GBM_Search_Tool",  "Modeling_Tool/Model/GBM_Search_Tool.py"),
    ("Modeling_Tool.Model.LRM_Tool",         "Modeling_Tool/Model/LRM_Tool.py"),
    ("Modeling_Tool.Eval.Evaluation_Tool", "Modeling_Tool/Eval/Evaluation_Tool.py"),
    ("Modeling_Tool.Eval.Model_Eval_Tool", "Modeling_Tool/Eval/Model_Eval_Tool.py"),
    ("Modeling_Tool.Eval.evaluate_model",  "Modeling_Tool/Eval/evaluate_model.py"),
    ("Modeling_Tool.Sample.Distribution_Adaptation", "Modeling_Tool/Sample/Distribution_Adaptation.py"),
    ("Modeling_Tool.Sample.Reject_Infer",            "Modeling_Tool/Sample/Reject_Infer.py"),
    ("Modeling_Tool.Sample.Sample_Split",            "Modeling_Tool/Sample/Sample_Split.py"),
    ("Modeling_Tool.Core.Binning_Tool",   "Modeling_Tool/Core/Binning_Tool.py"),
    ("Modeling_Tool.Core.kDataFrame",     "Modeling_Tool/Core/kDataFrame.py"),
    ("Modeling_Tool.Core.XOR_Encryptor",  "Modeling_Tool/Core/XOR_Encryptor.py"),
    ("Modeling_Tool.Core.Slope_Tool",     "Modeling_Tool/Core/Slope_Tool.py"),
]

extensions = [Extension(name=mod, sources=[src]) for mod, src in CLOSED_SOURCE_MODULES]
CLOSED_SOURCE_DOTTED = {mod for mod, _ in CLOSED_SOURCE_MODULES}

class build_py_strip_closed_source(_build_py):
    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        filtered = []
        for pkg, mod, filepath in modules:
            dotted = f"{pkg}.{mod}"
            if dotted in CLOSED_SOURCE_DOTTED:
                continue
            if mod.startswith("test_") or mod == "conftest":
                continue
            filtered.append((pkg, mod, filepath))
        return filtered

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
    packages=find_packages(exclude=["*.backup", "*.backup.*", "backup.*", "backup", "tests", "tests.*"]),
    package_data={
        "Modeling_Tool": ["*.pyi"],
        "Modeling_Tool.WOE": ["*.pyi"],
        "Modeling_Tool.Feature": ["*.pyi"],
        "Modeling_Tool.Model": ["*.pyi"],
        "Modeling_Tool.Eval": ["*.pyi"],
        "Modeling_Tool.Sample": ["*.pyi"],
        "Modeling_Tool.Core": ["*.pyi"],
    },
    ext_modules=cythonize(extensions, compiler_directives=CYTHON_DIRECTIVES, force=True, quiet=False),
    cmdclass={"build_py": build_py_strip_closed_source},
    zip_safe=False,
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
