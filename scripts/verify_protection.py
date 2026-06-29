#!/usr/bin/env python3
"""Compatibility wrapper for the retired closed-source protection verifier.

The project now ships Python sources directly, so the old Cython/stub/MANIFEST
parity checks no longer apply. This script remains as a no-op for any external
automation that still invokes it.
"""

from __future__ import annotations


def main() -> int:
    print("Closed-source protection checks retired: package is distributed as source.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
