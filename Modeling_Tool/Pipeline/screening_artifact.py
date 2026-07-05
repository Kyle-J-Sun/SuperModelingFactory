"""Handoff contract between feature validation and credit modeling pipelines."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

import pandas as pd

from Modeling_Tool.Feature.Weighted_Screen import WeightedScreenResult


def screen_result_to_summary(
    result: WeightedScreenResult,
    initial_features: list[str],
) -> dict[str, Any]:
    """Convert a screening result into CM-compatible ``feature_selection_summary``."""
    summary: dict[str, Any] = {"initial_features": list(initial_features)}
    if not result.psi_table.empty:
        psi = result.psi_table.copy()
        if "psi_ins_oos" in psi.columns:
            psi["psi"] = psi["psi_ins_oos"]
        elif "psi_max" in psi.columns:
            psi["psi"] = psi["psi_max"]
        summary["psi"] = psi
    if not result.iv_table.empty:
        iv = result.iv_table.copy()
        if "iv_weighted" in iv.columns:
            iv = iv.rename(columns={"iv_weighted": "iv"})
        summary["iv"] = iv
    if not result.corr_dropped.empty:
        summary["corr_dropped"] = result.corr_dropped
    summary["corr_features"] = list(result.selected_features)
    summary["screen_summary"] = result.summary
    summary["final_features"] = list(result.selected_features)
    return summary


@dataclass
class FeatureScreeningArtifact:
    selected_features: list[str]
    selection_summary: dict[str, Any]
    woe_artifacts: dict[str, Any] | None
    source: Literal["fvp", "cm", "standalone"]
    target_col: str
    weight_col: str | None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    created_at: str | None = None

    @classmethod
    def from_screen_result(
        cls,
        result: WeightedScreenResult,
        *,
        initial_features: list[str],
        target_col: str,
        weight_col: str | None,
        woe_artifacts: dict[str, Any] | None,
        source: Literal["fvp", "cm", "standalone"],
        config_snapshot: dict[str, Any] | None = None,
    ) -> FeatureScreeningArtifact:
        return cls(
            selected_features=list(result.selected_features),
            selection_summary=screen_result_to_summary(result, initial_features),
            woe_artifacts=woe_artifacts,
            source=source,
            target_col=target_col,
            weight_col=weight_col,
            config_snapshot=dict(config_snapshot or {}),
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @classmethod
    def from_fvp_result(
        cls,
        result: Any,
        *,
        target_col: str | None = None,
    ) -> FeatureScreeningArtifact:
        if getattr(result, "screening_artifact", None) is not None:
            return result.screening_artifact
        resolved_target = target_col
        if not resolved_target:
            summary = getattr(result, "selection_summary", {}) or {}
            resolved_target = summary.get("target_col")
        if not resolved_target:
            raise ValueError("target_col is required when building artifact from FVP result.")
        summary = dict(getattr(result, "selection_summary", {}) or {})
        config_snapshot = dict(summary.get("config_snapshot", {}) or {})
        return cls(
            selected_features=list(getattr(result, "selected_features", []) or []),
            selection_summary=summary,
            woe_artifacts=getattr(result, "woe_artifacts", None),
            source="fvp",
            target_col=str(resolved_target),
            weight_col=config_snapshot.get("weight_col"),
            config_snapshot=config_snapshot,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def validate_for_cm(self, *, target_col: str, weight_col: str | None = None) -> None:
        if self.target_col != target_col:
            raise ValueError(
                f"screening artifact target_col {self.target_col!r} does not match CM target_col {target_col!r}"
            )
        if self.weight_col != weight_col:
            raise ValueError(
                f"screening artifact weight_col {self.weight_col!r} does not match CM weight_col {weight_col!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.DataFrame):
                payload[key] = value.to_dict(orient="records")
        return payload


__all__ = [
    "FeatureScreeningArtifact",
    "screen_result_to_summary",
]