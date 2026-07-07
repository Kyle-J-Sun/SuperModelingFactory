"""Robustness helpers for feature screening and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SMFFailureLogger:
    """Small in-memory failure collector used for record-and-continue paths."""

    failures: list[dict[str, Any]] = field(default_factory=list)

    def record_and_continue(self, feat: str, exc: Exception, stage: str) -> dict[str, Any]:
        row = {
            "feature": feat,
            "stage": stage,
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        self.failures.append(row)
        return row

    def get_failures(self, stage: str | None = None) -> list[dict[str, Any]]:
        if stage is None:
            return list(self.failures)
        return [row for row in self.failures if row.get("stage") == stage]

    def clear(self, stage: str | None = None) -> None:
        if stage is None:
            self.failures.clear()
            return
        self.failures[:] = [row for row in self.failures if row.get("stage") != stage]


smf_logger = SMFFailureLogger()


def iv_guard(bad_w: float, good_w: float, eps: float = 1e-8) -> tuple[float, bool]:
    """Return an IV contribution and whether the bin is class-degenerate.

    Degenerate bins are intentionally excluded from IV sums. The epsilon is only
    used to protect against tiny negative floating point noise around zero.
    """

    bad = float(bad_w)
    good = float(good_w)
    if not np.isfinite(bad) or not np.isfinite(good):
        return 0.0, True
    if bad <= eps or good <= eps:
        return 0.0, True
    return float((bad - good) * np.log(bad / good)), False
