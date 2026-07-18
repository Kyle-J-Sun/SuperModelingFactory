"""Post-correlation selection gates (audit items G03/G04/G05/G06).

These gates run after the classic PSI -> IV -> corr sequence inside every
``feature_screen`` path, immediately before the final summary row:

- VIF gate (G06): iterative drop-highest multicollinearity filter.
- Group-stability gate (G03): monthly (or any group-dim) IV floor / CV cap /
  direction-consistency floor, fed by FVP-precomputed evidence.
- Multi-target gate (G04): joint IV/direction rules across several labels.
- Truncation gate (G05): hard top-N by a ranking metric with a stable
  tie-breaker.

Evidence for G03/G04 cannot be derived inside the screens (they receive a
single target and INS-only splits), so FVP builds a ``SelectionEvidence``
with lazy closures priced only on the post-corr survivor set.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight

from .Weighted_Screen import _apply_stage_keep, _summary_row


def point_biserial_direction(
    x: pd.Series, y: pd.Series, sample_weight: np.ndarray | None = None
) -> int:
    """Sign of the point-biserial association between a numeric feature and a
    binary target: +1 when higher x means higher bad rate (WOE increasing),
    -1 for the opposite, 0 when undefined. The single shared definition of
    "direction" for G03/G04.

    ``sample_weight`` is positional-aligned with the passed ``x``/``y``
    (before NaN masking). A class whose total weight is zero — like a class
    with no rows — and a weighted tie both return 0, mirroring the
    unweighted NaN/tie semantics."""
    xv = pd.to_numeric(x, errors="coerce")
    yv = pd.to_numeric(y, errors="coerce")
    mask = xv.notna() & yv.notna()
    if not bool(mask.any()):
        return 0
    if sample_weight is None:
        xv = xv[mask]
        yv = yv[mask]
        mean_bad = xv[yv == 1].mean()
        mean_good = xv[yv == 0].mean()
    else:
        mask_np = mask.to_numpy()
        w = np.asarray(sample_weight, dtype=float)[mask_np]
        xa = xv.to_numpy(dtype=float)[mask_np]
        ya = yv.to_numpy(dtype=float)[mask_np]
        w_bad = w[ya == 1]
        w_good = w[ya == 0]
        mean_bad = (
            float(np.average(xa[ya == 1], weights=w_bad)) if w_bad.sum() > 0 else float("nan")
        )
        mean_good = (
            float(np.average(xa[ya == 0], weights=w_good)) if w_good.sum() > 0 else float("nan")
        )
    if pd.isna(mean_bad) or pd.isna(mean_good) or mean_bad == mean_good:
        return 0
    return 1 if mean_bad > mean_good else -1


@dataclass
class SelectionEvidence:
    """FVP-precomputed evidence for the G03/G04 gates.

    per_group_iv_fn(features) -> DataFrame[var, group, n, iv, direction]
    per_target_fn(features)   -> DataFrame[var, target, iv, direction, status]
    Both closures are lazy so they only price the post-corr survivor set.
    """

    group_dims: list[str] = field(default_factory=list)
    scope: str = "ins"
    per_group_iv_fn: Callable[[list[str]], pd.DataFrame] | None = None
    per_target_fn: Callable[[list[str]], pd.DataFrame] | None = None
    min_group_n_default: int = 0


def _config_needs_evidence(config: Any) -> list[str]:
    """Names of configured gate fields that require SelectionEvidence."""
    needed = []
    if any(
        getattr(config, name, None) is not None
        for name in ("monthly_iv_min", "monthly_iv_cv_max", "direction_consistency_min")
    ):
        needed.append("group_stability (monthly_iv_min/monthly_iv_cv_max/direction_consistency_min)")
    if getattr(config, "target_rules", None) is not None:
        needed.append("multi_target (target_rules)")
    return needed


def apply_vif_stage(
    ins: pd.DataFrame,
    current: list[str],
    config: Any,
    iv_map: dict[str, float],
    summary_rows: list[dict],
    dropped_rows: list[dict],
    stage_tables: dict[str, pd.DataFrame],
    *,
    weight_col: str | None,
    on_empty_stage: str,
    woe_frame_fn: Callable[[list[str]], pd.DataFrame] | None = None,
) -> list[str]:
    """G06: iteratively drop the highest-VIF feature until all VIFs fall
    below ``vif_threshold`` or only ``vif_min_features`` remain.

    G06-2: the VIF matrix basis is selectable. ``vif_use_woe_bins=True``
    computes VIF on the WOE-encoded INS view (``woe_frame_fn``; all-numeric,
    matches the LR design-matrix collinearity semantics). The legacy raw-value
    basis excludes non-numeric survivors from the matrix — raw string columns
    used to crash statsmodels — keeping them in the selection untouched.
    """
    if not getattr(config, "vif_enabled", False):
        return current

    tie_metric = str(getattr(config, "vif_tie_break_metric", "iv"))
    if tie_metric != "iv":
        raise ValueError(
            "vif_tie_break_metric currently supports only 'iv'; "
            f"got {tie_metric!r}"
        )

    if len(current) <= max(2, int(config.vif_min_features)):
        if len(current) <= int(config.vif_min_features):
            summary_rows.append(_summary_row(
                "vif", len(current), len(current), config.vif_threshold, weight_col,
                note="skipped_at_floor",
            ))
        return current

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "vif_enabled=True requires statsmodels (optional extra). Install it "
            "with: pip install \"SuperModelingFactory[stats]\""
        ) from exc

    from Modeling_Tool.Model.LRM_Tool import FeatureSelectionAnalyzer

    analyzer = FeatureSelectionAnalyzer()
    threshold = float(config.vif_threshold)
    floor = int(config.vif_min_features)
    sample_weight = (
        resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))
        if weight_col is not None
        else None
    )
    excluded: list[str] = []
    raw_vif_cast_columns: list[str] = []
    if bool(getattr(config, "vif_use_woe_bins", False)):
        if woe_frame_fn is None:
            raise ValueError(
                "vif_use_woe_bins=True requires a fitted screening WOE engine "
                "(enable psi_use_woe_bins/iv_use_woe_bins/corr_use_woe_bins or "
                "pass a prefit engine) so the VIF matrix can be WOE-encoded."
            )
        base = woe_frame_fn(list(current))
        if not isinstance(base, pd.DataFrame):
            raise ValueError(
                "vif_use_woe_bins=True requires woe_frame_fn to return a pandas DataFrame."
            )
        missing = [name for name in current if name not in base.columns]
        if missing:
            raise ValueError(
                "vif_use_woe_bins=True could not produce a VIF column for "
                f"{missing!r}. Refit the screening WOE engine or supply a "
                "prefit engine that covers every surviving feature."
            )
        non_numeric = [name for name in current if not pd.api.types.is_numeric_dtype(base[name])]
        if non_numeric:
            raise ValueError(
                "vif_use_woe_bins=True requires numeric WOE columns; got "
                f"non-numeric output for {non_numeric!r}."
            )
        survivors = list(current)
    else:
        survivors = [c for c in current if pd.api.types.is_numeric_dtype(ins[c])]
        excluded = [c for c in current if c not in set(survivors)]
        if excluded:
            preview = ", ".join(excluded[:5]) + ("..." if len(excluded) > 5 else "")
            warnings.warn(
                f"vif gate: {len(excluded)} non-numeric feature(s) excluded from "
                f"the raw-value VIF matrix and kept in the selection: {preview}. "
                f"Raw non-numeric columns crash statsmodels OLS; set "
                f"vif_use_woe_bins=True to include them via the WOE-encoded "
                f"matrix instead.",
                UserWarning,
                stacklevel=3,
            )
        base = ins
        if len(survivors) <= max(2, floor):
            summary_rows.append(_summary_row(
                "vif", len(current), len(current), threshold, weight_col,
                note="skipped_insufficient_numeric",
                n_excluded_non_numeric=len(excluded),
                excluded_non_numeric=list(excluded),
            ))
            return current
        # pandas nullable numeric/boolean and bool columns are logically
        # numeric but produce object arrays in statsmodels.  Convert only
        # those VIF inputs; a normal NumPy numeric frame stays byte-identical.
        base = ins[survivors]
        for name in survivors:
            dtype = base[name].dtype
            if (
                pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_extension_array_dtype(dtype)
            ):
                if not raw_vif_cast_columns:
                    base = base.copy()
                base[name] = pd.to_numeric(base[name], errors="raise").astype(float)
                raw_vif_cast_columns.append(name)
    iteration_rows: list[dict] = []
    for iteration in range(len(current)):
        if len(survivors) <= floor:
            break
        if sample_weight is None:
            # Preserve the historical unweighted call path byte-for-byte.
            vif_table = analyzer.compute_vif(base[survivors])
        else:
            vif_table = analyzer.compute_vif(
                base[survivors], sample_weight=sample_weight,
            )
        vif_table = vif_table.sort_values("VIF", ascending=False).reset_index(drop=True)
        worst = vif_table.iloc[0]
        if not np.isfinite(worst["VIF"]) or worst["VIF"] > threshold:
            # Ties on VIF break toward the LOWER tie-break metric (weaker
            # feature drops first), then lexicographic name for determinism.
            top = vif_table[
                (~np.isfinite(vif_table["VIF"])) if not np.isfinite(worst["VIF"])
                else (vif_table["VIF"] == worst["VIF"])
            ]
            drop = sorted(
                top["feature"].tolist(),
                key=lambda name: (iv_map.get(name, 0.0), name),
            )[0]
            drop_vif = float(vif_table.loc[vif_table["feature"] == drop, "VIF"].iloc[0])
            survivors.remove(drop)
            row = {
                "iteration": iteration, "dropped": drop, "vif": drop_vif,
                "n_remaining": len(survivors) + len(excluded),
            }
            if excluded:
                row["n_vif_features"] = len(survivors)
                row["n_excluded_non_numeric"] = len(excluded)
            if raw_vif_cast_columns:
                row["cast_vif_columns"] = list(raw_vif_cast_columns)
            iteration_rows.append(row)
            dropped_rows.append({
                "var": drop, "stage": "vif", "metric": "vif",
                "value": drop_vif, "threshold": threshold, "reason": "vif_above_threshold",
            })
        else:
            break
    if iteration_rows:
        stage_tables["vif"] = pd.DataFrame(iteration_rows)
    kept = _apply_stage_keep(
        current, survivors + excluded, "vif", summary_rows,
        on_empty_stage=on_empty_stage, weight_col=weight_col,
        threshold=threshold, intersect=True,
    )
    audit: dict[str, Any] = {}
    if excluded:
        # Warnings are intentionally supplemental: application-level warning
        # filters can hide them, whereas the selection summary is an artifact.
        audit.update({
            "note": "excluded_non_numeric",
            "n_excluded_non_numeric": len(excluded),
            "excluded_non_numeric": list(excluded),
        })
    if raw_vif_cast_columns:
        audit["cast_vif_columns"] = list(raw_vif_cast_columns)
    summary_rows.append(_summary_row(
        "vif", len(current), len(kept), threshold, weight_col, **audit,
    ))
    return kept


def apply_group_stability_stage(
    current: list[str],
    config: Any,
    evidence: SelectionEvidence,
    summary_rows: list[dict],
    dropped_rows: list[dict],
    stage_tables: dict[str, pd.DataFrame],
    *,
    weight_col: str | None,
    on_empty_stage: str,
) -> list[str]:
    """G03: per-group (e.g. monthly) IV floor, IV CV cap, and direction
    consistency floor. Features with too few eligible groups are handled per
    ``insufficient_group_policy``."""
    thresholds_active = any(
        getattr(config, name, None) is not None
        for name in ("monthly_iv_min", "monthly_iv_cv_max", "direction_consistency_min")
    )
    if not thresholds_active or not current:
        return current
    frame = evidence.per_group_iv_fn(list(current)) if evidence.per_group_iv_fn else pd.DataFrame()
    min_group_n = int(getattr(config, "min_group_n", None) or evidence.min_group_n_default or 0)
    policy = str(getattr(config, "insufficient_group_policy", "keep_warn") or "keep_warn")

    keep: list[str] = []
    insufficient: list[str] = []
    detail_rows: list[dict] = []
    for var in current:
        sub = frame[frame["var"] == var] if len(frame) else pd.DataFrame()
        if len(sub) and min_group_n:
            sub = sub[sub["n"] >= min_group_n]
        if len(sub) < 2:
            insufficient.append(var)
            continue
        ivs = sub["iv"].astype(float)
        directions = sub["direction"].astype(int)
        reasons = []
        iv_min_thr = getattr(config, "monthly_iv_min", None)
        if iv_min_thr is not None and float(ivs.min()) < float(iv_min_thr):
            reasons.append(("monthly_iv_min", float(ivs.min()), float(iv_min_thr)))
        cv_thr = getattr(config, "monthly_iv_cv_max", None)
        if cv_thr is not None:
            mean_iv = float(ivs.mean())
            cv = float(ivs.std(ddof=0) / mean_iv) if mean_iv else float("inf")
            if cv > float(cv_thr):
                reasons.append(("monthly_iv_cv_max", cv, float(cv_thr)))
        dir_thr = getattr(config, "direction_consistency_min", None)
        if dir_thr is not None:
            non_zero = directions[directions != 0]
            if len(non_zero):
                modal = non_zero.mode().iloc[0]
                consistency = float((non_zero == modal).mean())
            else:
                consistency = 0.0
            if consistency < float(dir_thr):
                reasons.append(("direction_consistency_min", consistency, float(dir_thr)))
        detail_rows.extend(
            {"var": var, "group_metric": name, "value": value, "threshold": thr}
            for name, value, thr in reasons
        )
        if reasons:
            name, value, thr = reasons[0]
            dropped_rows.append({
                "var": var, "stage": "group_stability", "metric": name,
                "value": value, "threshold": thr, "reason": name,
            })
        else:
            keep.append(var)

    if insufficient:
        if policy == "raise":
            raise ValueError(
                f"group_stability: {len(insufficient)} feature(s) have fewer than 2 "
                f"eligible groups (min_group_n={min_group_n}), e.g. {insufficient[:5]}. "
                f"Set insufficient_group_policy='keep_warn'/'drop' to proceed."
            )
        if policy == "drop":
            for var in insufficient:
                dropped_rows.append({
                    "var": var, "stage": "group_stability", "metric": "eligible_groups",
                    "value": None, "threshold": 2, "reason": "insufficient_groups_dropped",
                })
        else:  # keep_warn
            warnings.warn(
                f"group_stability: {len(insufficient)} feature(s) have fewer than 2 "
                f"eligible groups and were kept unchecked (insufficient_group_policy="
                f"'keep_warn'), e.g. {insufficient[:5]}.",
                UserWarning,
                stacklevel=3,
            )
            keep.extend(insufficient)

    if len(frame):
        stage_tables["group_stability"] = frame
    if detail_rows:
        stage_tables["group_stability_violations"] = pd.DataFrame(detail_rows)
    kept = _apply_stage_keep(
        current, keep, "group_stability", summary_rows,
        on_empty_stage=on_empty_stage, weight_col=weight_col, intersect=True,
    )
    summary_rows.append(_summary_row(
        "group_stability", len(current), len(kept), None, weight_col,
        monthly_iv_min=getattr(config, "monthly_iv_min", None),
        monthly_iv_cv_max=getattr(config, "monthly_iv_cv_max", None),
        direction_consistency_min=getattr(config, "direction_consistency_min", None),
    ))
    return kept


def apply_multi_target_stage(
    current: list[str],
    config: Any,
    evidence: SelectionEvidence,
    summary_rows: list[dict],
    dropped_rows: list[dict],
    stage_tables: dict[str, pd.DataFrame],
    *,
    weight_col: str | None,
    on_empty_stage: str,
) -> list[str]:
    """G04: joint gate across several labels — a feature must pass its
    per-target IV range / direction alignment on all / any / >=K targets."""
    rule = getattr(config, "target_rules", None)
    if rule is None or not current:
        return current
    if rule not in {"all", "any", "min_pass_count"}:
        raise ValueError(
            f"target_rules must be 'all'/'any'/'min_pass_count'; got {rule!r}"
        )
    frame = evidence.per_target_fn(list(current)) if evidence.per_target_fn else pd.DataFrame()
    if not len(frame):
        warnings.warn(
            "multi_target gate configured but no per-target evidence is "
            "available; keeping all features unchecked.",
            UserWarning,
            stacklevel=3,
        )
        return current
    iv_range = getattr(config, "per_target_iv_range", None)
    ref_target = getattr(config, "direction_reference_target", None)
    min_pass = int(getattr(config, "min_pass_count", None) or 1)

    def _range_for(target: str):
        if iv_range is None:
            return None
        if isinstance(iv_range, dict):
            return iv_range.get(target)
        return tuple(iv_range)

    ref_directions: dict[str, int] = {}
    if ref_target is not None:
        ref_rows = frame[frame["target"] == ref_target]
        ref_directions = dict(zip(ref_rows["var"], ref_rows["direction"]))

    keep: list[str] = []
    for var in current:
        sub = frame[(frame["var"] == var) & (frame["status"] == "ok")]
        n_targets = len(sub)
        if n_targets == 0:
            warnings.warn(
                f"multi_target: no usable target evidence for {var!r}; kept unchecked.",
                UserWarning,
                stacklevel=3,
            )
            keep.append(var)
            continue
        passes = 0
        for _, row in sub.iterrows():
            ok = True
            rng = _range_for(str(row["target"]))
            if rng is not None:
                low, high = rng
                iv_value = float(row["iv"])
                if (low is not None and iv_value < low) or (high is not None and iv_value > high):
                    ok = False
            if ok and ref_target is not None:
                ref_dir = ref_directions.get(var, 0)
                if ref_dir != 0 and int(row["direction"]) != 0 and int(row["direction"]) != ref_dir:
                    ok = False
            passes += int(ok)
        required = n_targets if rule == "all" else (1 if rule == "any" else min_pass)
        if passes >= required:
            keep.append(var)
        else:
            dropped_rows.append({
                "var": var, "stage": "multi_target", "metric": "pass_count",
                "value": passes, "threshold": required, "reason": f"target_rules_{rule}",
            })

    stage_tables["multi_target"] = frame
    kept = _apply_stage_keep(
        current, keep, "multi_target", summary_rows,
        on_empty_stage=on_empty_stage, weight_col=weight_col, intersect=True,
    )
    summary_rows.append(_summary_row(
        "multi_target", len(current), len(kept), None, weight_col,
        target_rules=rule, min_pass_count=getattr(config, "min_pass_count", None),
    ))
    return kept


def apply_truncation_stage(
    current: list[str],
    config: Any,
    iv_map: dict[str, float],
    summary_rows: list[dict],
    dropped_rows: list[dict],
    stage_tables: dict[str, pd.DataFrame],
    *,
    weight_col: str | None,
) -> list[str]:
    """G05: hard cap on the final feature count, ranked by ranking_metric
    with a deterministic tie-breaker. Runs last; never backfills."""
    cap = getattr(config, "max_selected_features", None)
    floor = getattr(config, "min_selected_features", None)
    if cap is None and floor is None:
        return current
    ranked = list(current)
    if cap is not None and len(current) > int(cap):
        metric = str(getattr(config, "ranking_metric", "iv") or "iv")
        if metric != "iv":
            raise ValueError(
                f"ranking_metric {metric!r} is not available in this release; use 'iv'."
            )
        tie = str(getattr(config, "tie_breaker", "name") or "name")
        ranked = sorted(
            current,
            key=lambda name: (-iv_map.get(name, 0.0), name if tie == "name" else name),
        )
        kept_set = ranked[: int(cap)]
        truncated = [name for name in ranked[int(cap):]]
        for name in truncated:
            dropped_rows.append({
                "var": name, "stage": "truncation", "metric": metric,
                "value": iv_map.get(name, 0.0), "threshold": int(cap),
                "reason": "max_selected_features",
            })
        stage_tables["truncation"] = pd.DataFrame({
            "var": ranked,
            "rank": range(1, len(ranked) + 1),
            metric: [iv_map.get(name, 0.0) for name in ranked],
            "kept": [name in set(kept_set) for name in ranked],
        })
        new_current = [name for name in current if name in set(kept_set)]
        summary_rows.append(_summary_row(
            "truncation", len(current), len(new_current), int(cap), weight_col,
            ranking_metric=metric,
        ))
        current = new_current
    if floor is not None and len(current) < int(floor):
        warnings.warn(
            f"truncation: only {len(current)} feature(s) survive the gates, below "
            f"min_selected_features={int(floor)}. No backfill is performed — "
            f"previously-dropped features stay dropped (audit causality).",
            UserWarning,
            stacklevel=3,
        )
        summary_rows.append(_summary_row(
            "truncation_fallback", len(current), len(current), int(floor), weight_col,
        ))
    return current


def apply_post_corr_gates(
    ins: pd.DataFrame,
    current: list[str],
    config: Any,
    evidence: SelectionEvidence | None,
    iv_map: dict[str, float],
    summary_rows: list[dict],
    dropped_rows: list[dict],
    stage_tables: dict[str, pd.DataFrame],
    *,
    weight_col: str | None,
    on_empty_stage: str,
    woe_frame_fn: Callable[[list[str]], pd.DataFrame] | None = None,
) -> list[str]:
    """Orchestrate the post-corr gates: vif -> group_stability ->
    multi_target -> truncation. Raises when G03/G04 thresholds are set but no
    SelectionEvidence was provided (CMP path — FVP-only gates this release)."""
    needed = _config_needs_evidence(config)
    if needed and evidence is None:
        raise ValueError(
            f"Selection gates {needed} require SelectionEvidence, which only "
            f"FeatureValidationPipeline provides in this release. Drop those "
            f"thresholds on this path or run the screen through FVP."
        )
    current = apply_vif_stage(
        ins, current, config, iv_map, summary_rows, dropped_rows, stage_tables,
        weight_col=weight_col, on_empty_stage=on_empty_stage,
        woe_frame_fn=woe_frame_fn,
    )
    if evidence is not None:
        current = apply_group_stability_stage(
            current, config, evidence, summary_rows, dropped_rows, stage_tables,
            weight_col=weight_col, on_empty_stage=on_empty_stage,
        )
        current = apply_multi_target_stage(
            current, config, evidence, summary_rows, dropped_rows, stage_tables,
            weight_col=weight_col, on_empty_stage=on_empty_stage,
        )
    current = apply_truncation_stage(
        current, config, iv_map, summary_rows, dropped_rows, stage_tables,
        weight_col=weight_col,
    )
    return current


__all__ = [
    "SelectionEvidence",
    "point_biserial_direction",
    "apply_post_corr_gates",
    "apply_vif_stage",
    "apply_group_stability_stage",
    "apply_multi_target_stage",
    "apply_truncation_stage",
]
