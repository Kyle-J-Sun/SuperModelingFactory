#!/usr/bin/env python3
"""Legacy stub-generation entry point.

SuperModelingFactory now ships its Python sources directly. Historical `.pyi`
stubs may remain in the tree for editor support, but package builds no longer
require generating stubs to hide implementation code.
"""

from __future__ import annotations


def main() -> int:
    print("SuperModelingFactory is open-source; stub generation is no longer required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
