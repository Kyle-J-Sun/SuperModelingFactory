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
    """Convert a screening result into CM-compatible ``feature_selection_summary``.

    Parameters
    ----------
    result : WeightedScreenResult
        Result of a weighted feature screen.
    initial_features : list of str
        The features that entered the screen.

    Returns
    -------
    dict
        Always ``initial_features``, ``corr_features``, ``screen_summary`` and ``final_features`` (the last two lists are
        the selected features). Present only when not empty: ``psi`` (with a ``psi`` column taken from ``psi_ins_oos`` or
        ``psi_max``), ``iv`` (``iv_weighted`` renamed to ``iv``), ``corr_dropped``, ``missing_rate``,
        ``missing_rate_dropped``, ``dropped_detail`` and ``stage_tables``.
    """
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
    if not result.missing_rate_table.empty:
        summary["missing_rate"] = result.missing_rate_table
    if not result.missing_rate_dropped.empty:
        summary["missing_rate_dropped"] = result.missing_rate_dropped
    summary["corr_features"] = list(result.selected_features)
    summary["screen_summary"] = result.summary
    summary["final_features"] = list(result.selected_features)
    dropped_detail = getattr(result, "dropped_detail", None)
    if dropped_detail is not None and len(dropped_detail):
        summary["dropped_detail"] = dropped_detail
    stage_tables = getattr(result, "stage_tables", None)
    if stage_tables:
        summary["stage_tables"] = dict(stage_tables)
    return summary


def woe_artifacts_from_screen_result(
    result: WeightedScreenResult,
    target_col: str,
) -> dict[str, Any] | None:
    """Wrap a screen-fitted WOE engine into the CM reuse contract (G00).

    The returned dict matches what CreditModelPipeline._reuse_screening_woe
    consumes: ``by_target[target] = {engine, adapter, features}`` plus a
    top-level ``woe_table`` and the ``engine_meta`` of the screen.

    Parameters
    ----------
    result : WeightedScreenResult
        Result of a weighted feature screen; its ``woe_engine`` and ``woe_engine_meta`` attributes are used.
    target_col : str
        Target the engine was fitted for; it becomes the key of ``by_target``.

    Returns
    -------
    dict or None
        The reuse payload, or None when the screen did not attach a reusable engine (for example WOE_Master screens
        attach table-only metadata).
    """
    engine = getattr(result, "woe_engine", None)
    meta = dict(getattr(result, "woe_engine_meta", {}) or {})
    if engine is None:
        return None
    from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

    adapter = as_woe_engine(engine)
    features = list(meta.get("fit_features") or result.selected_features)
    try:
        woe_table = adapter.get_woe_table(features)
    except Exception:
        woe_table = pd.DataFrame()
    return {
        "by_target": {
            target_col: {
                "engine": engine,
                "adapter": adapter,
                "features": features,
            }
        },
        "woe_table": woe_table,
        "engine_meta": meta,
    }


@dataclass
class FeatureScreeningArtifact:
    """Handoff contract between feature validation and credit modeling.

    Parameters
    ----------
    selected_features : list of str
        Features that survived the screening.
    selection_summary : dict
        The screening summary in the layout of ``screen_result_to_summary``.
    woe_artifacts : dict or None
        Reusable WOE engines and tables, or None when the screen did not keep any.
    source : {"fvp", "cm", "standalone"}
        Which pipeline produced the artifact.
    target_col : str
        Target the screening was run for.
    weight_col : str or None
        Sample-weight column used by the screening, or None.
    config_snapshot : dict, default empty dict
        Copy of the configuration that produced the screening.
    created_at : str or None, default None
        UTC creation time in ISO format.
    """

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
        """Build an artifact from the result of a weighted feature screen.

        Parameters
        ----------
        result : WeightedScreenResult
            Result of the screen.
        initial_features : list of str
            The features that entered the screen.
        target_col : str
            Target the screen was run for.
        weight_col : str or None
            Sample-weight column used by the screen.
        woe_artifacts : dict or None
            WOE artifacts to store; when None or empty, the engine that the screen itself fitted is wrapped (if there is one).
        source : {"fvp", "cm", "standalone"}
            Which pipeline produced the result.
        config_snapshot : dict or None, default None
            Copy of the configuration to store.

        Returns
        -------
        FeatureScreeningArtifact
        """
        if not woe_artifacts:
            # G00: reuse the engine the screen itself fitted, when available (an empty dict, which a run with
            # woe_enabled=False passes, carries nothing to reuse either).
            woe_artifacts = woe_artifacts_from_screen_result(result, target_col)
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
        weight_col: str | None = None,
    ) -> FeatureScreeningArtifact:
        """Build an artifact from a feature-validation pipeline result.

        When the result already carries a ``screening_artifact`` it is returned unchanged. Otherwise the artifact is built
        from the result's ``selection_summary`` and ``config_snapshot``. If the pipeline ran with selection switched off
        and selected nothing, the configured ``new_feature_cols`` become the selected features.

        Parameters
        ----------
        result : FeatureValidationPipelineResult
            Result of a feature-validation run.
        target_col : str or None, default None
            Target column; when None it is read from the summary, then from the config snapshot (``target_col``, then the
            first of ``target_cols``).
        weight_col : str or None, default None
            Weight column. A ``weight_col`` entry in the config snapshot takes precedence over this argument.

        Returns
        -------
        FeatureScreeningArtifact

        Raises
        ------
        ValueError
            If no target column can be determined.
        """
        if getattr(result, "screening_artifact", None) is not None:
            return result.screening_artifact
        summary = dict(getattr(result, "selection_summary", {}) or {})
        summary_snapshot = dict(summary.get("config_snapshot", {}) or {})
        result_snapshot = dict(getattr(result, "config_snapshot", {}) or {})
        config_snapshot = {**summary_snapshot, **result_snapshot}
        resolved_target = target_col
        if not resolved_target:
            resolved_target = summary.get("target_col")
        if not resolved_target:
            resolved_target = config_snapshot.get("target_col")
        if not resolved_target:
            target_cols = list(config_snapshot.get("target_cols") or [])
            resolved_target = target_cols[0] if target_cols else None
        if not resolved_target:
            raise ValueError("target_col is required when building artifact from FVP result.")
        selected_features = list(getattr(result, "selected_features", []) or [])
        if not selected_features and config_snapshot.get("selection_enabled") is False:
            selected_features = list(config_snapshot.get("new_feature_cols") or [])
        if not summary and config_snapshot.get("selection_enabled") is False:
            summary = {
                "initial_features": list(selected_features),
                "final_features": list(selected_features),
                "target_col": str(resolved_target),
                "selection_skipped": True,
                "config_snapshot": config_snapshot,
            }
        resolved_weight_col = (
            config_snapshot["weight_col"]
            if "weight_col" in config_snapshot
            else weight_col
        )
        return cls(
            selected_features=selected_features,
            selection_summary=summary,
            woe_artifacts=getattr(result, "woe_artifacts", None),
            source="fvp",
            target_col=str(resolved_target),
            weight_col=resolved_weight_col,
            config_snapshot=config_snapshot,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    def validate_for_cm(self, *, target_col: str, weight_col: str | None = None) -> None:
        """Check that the artifact was built for the same target and weight columns as a credit-model run.

        Parameters
        ----------
        target_col : str
            Target column of the credit-model configuration.
        weight_col : str or None, default None
            Weight column of the credit-model configuration.

        Raises
        ------
        ValueError
            If the artifact's ``target_col`` or ``weight_col`` differs.
        """
        if self.target_col != target_col:
            raise ValueError(
                f"screening artifact target_col {self.target_col!r} does not match CM target_col {target_col!r}"
            )
        if self.weight_col != weight_col:
            raise ValueError(
                f"screening artifact weight_col {self.weight_col!r} does not match CM weight_col {weight_col!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the artifact as a dictionary; top-level DataFrame values become lists of records."""
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.DataFrame):
                payload[key] = value.to_dict(orient="records")
        return payload


__all__ = [
    "FeatureScreeningArtifact",
    "screen_result_to_summary",
    "woe_artifacts_from_screen_result",
]
