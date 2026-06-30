from __future__ import annotations

import os
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent


def _read(path: str) -> str:
    file_path = ROOT / path
    return file_path.read_text(encoding="utf-8") if file_path.exists() else ""


setup(
    name="SuperModelingFactory",
    version=os.environ.get("SMF_VERSION", "0.2.1"),
    description="Credit risk modeling factory: WOE binning, scorecards, LightGBM, Excel reporting.",
    long_description=_read("README.md"),
    long_description_content_type="text/markdown",
    author="Kyle Sun",
    author_email="jingkai.sun20@alumni.imperial.ac.uk",
    url="https://github.com/Kyle-J-Sun/SuperModelingFactory",
    license="BUSL-1.1",
    python_requires=">=3.10",
    packages=find_packages(
        exclude=[
            "*.backup",
            "*.backup.*",
            "backup.*",
            "backup",
            "tests",
            "tests.*",
        ],
    ),
    package_data={
        "Modeling_Tool.WOE": ["*.pyi"],
        "Modeling_Tool.Feature": ["*.pyi"],
        "Modeling_Tool.Model": ["*.pyi"],
        "Modeling_Tool.Eval": ["*.pyi"],
        "Modeling_Tool.Sample": ["*.pyi"],
        "Modeling_Tool.Core": ["*.pyi"],
    },
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: Other/Proprietary License",
    ],
)
