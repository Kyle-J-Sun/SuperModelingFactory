"""Unified feature screening: PSI -> IV -> correlation dedup."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
from Modeling_Tool._utils.sentinels import SMF_MISSING_BIN

from .Weighted_Screen import (
    WeightedScreenResult,
    _apply_missing_rate_stage,
    _apply_stage_keep,
    _corr_dedup_weighted,
    _legacy_unweighted_screen,
    _psi_from_distributions,
    _resolve_splits,
    _summary_row,
    _weighted_bin_distribution,
    _weighted_corr_for_screen,
    _weighted_iv_from_assigned_bins,
    _weighted_screen_impl,
)

FeatureScreenResult = WeightedScreenResult

_MONOTONE_INIT_KEYS = frozenset({
    "n_init_bins",
    "min_bin_size",
    "min_n_bins",
    "eps",
    "missing_woe",
    "special_values",
    "bin_label_decimals",
})
_MONOTONE_FIT_KEYS = frozenset({"chi2_binning", "chi2_p", "chi2_init_size", "n_jobs"})


@dataclass
class FeatureScreenConfig:
    psi_enabled: bool = True
    psi_threshold: float = 0.2
    psi_compare_splits: list[str] = field(default_factory=lambda: ["oos"])
    psi_buckets: int = 10
    psi_use_woe_bins: bool = False
    iv_enabled: bool = True
    iv_threshold: float = 0.02
    iv_bins: int = 10
    iv_min_bin_prop: float = 0.05
    iv_equal_freq: bool = True
    iv_use_woe_bins: bool = False
    corr_enabled: bool = True
    corr_threshold: float = 0.75
    corr_max_iterations: int = 10
    corr_use_woe_bins: bool = False
    corr_nan_policy: Literal["pairwise", "median_fill", "raise"] = "pairwise"
    corr_block_size: int = 256
    on_empty_stage: Literal["keep_all_warn", "raise"] = "keep_all_warn"
    missing_rate_threshold: float | None = None
    missing_rate_ref: float | int = -999999
    woe_engine: str = "equal_freq"
    woe_fit_query: str | None = None
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}
    )
    categorical_features: list[str] | None = None
    plot_path: str | None = None
    plot_outputs: bool = False
    content: float = 1e-6
    precision: int = 5
    min_bin_prop: float = 0.05


def screen_config_from_mapping(
    mapping: Mapping[str, Any] | None,
    *,
    woe_engine: str | None = None,
    woe_fit_query: str | None = None,
    woe_params: Mapping[str, Any] | None = None,
    monotone_woe_params: Mapping[str, Any] | None = None,
    plot_path: str | None = None,
    plot_outputs: bool = False,
) -> FeatureScreenConfig:
    """Build ``FeatureScreenConfig`` from a CM-style ``feature_selection`` dict."""
    cfg = dict(mapping or {})
    iv_nbins = int(cfg.get("iv_nbins", cfg.get("iv_bins", 10)))
    return FeatureScreenConfig(
        psi_enabled=bool(cfg.get("psi_enabled", True)),
        psi_threshold=float(cfg.get("psi_threshold", 0.2)),
        psi_compare_splits=list(cfg.get("psi_compare_splits", ["oos"])),
        psi_buckets=int(cfg.get("psi_buckets", iv_nbins)),
        psi_use_woe_bins=bool(cfg.get("psi_use_woe_bins", False)),
        iv_enabled=bool(cfg.get("iv_enabled", True)),
        iv_threshold=float(cfg.get("iv_threshold", 0.02)),
        iv_bins=iv_nbins,
        iv_min_bin_prop=float(cfg.get("iv_min_bin_prop", 0.05)),
        iv_equal_freq=bool(cfg.get("iv_equal_freq", True)),
        iv_use_woe_bins=bool(cfg.get("iv_use_woe_bins", False)),
        corr_enabled=bool(cfg.get("corr_enabled", True)),
        corr_threshold=float(cfg.get("corr_threshold", 0.75)),
        corr_max_iterations=int(cfg.get("corr_max_iterations", 10)),
        corr_use_woe_bins=bool(cfg.get("corr_use_woe_bins", False)),
        corr_nan_policy=str(cfg.get("corr_nan_policy", "pairwise")),  # type: ignore[arg-type]
        corr_block_size=int(cfg.get("corr_block_size", 256)),
        on_empty_stage=str(cfg.get("on_empty_stage", "keep_all_warn")),  # type: ignore[arg-type]
        missing_rate_threshold=cfg.get("missing_rate_threshold"),
        missing_rate_ref=cfg.get("missing_rate_ref", -999999),
        woe_engine=str(woe_engine or cfg.get("woe_engine", "equal_freq")),
        woe_fit_query=woe_fit_query if woe_fit_query is not None else cfg.get("woe_fit_query"),
        woe_params=dict(FeatureScreenConfig().woe_params | dict(woe_params or cfg.get("woe_params") or {})),
        monotone_woe_params=dict(
            FeatureScreenConfig().monotone_woe_params | dict(monotone_woe_params or cfg.get("monotone_woe_params") or {})
        ),
        categorical_features=list(cfg["categorical_features"]) if cfg.get("categorical_features") else None,
        plot_path=plot_path,
        plot_outputs=plot_outputs,
        content=float(cfg.get("content", 1e-6)),
        precision=int(cfg.get("precision", 5)),
        min_bin_prop=float(cfg.get("min_bin_prop", cfg.get("iv_min_bin_prop", 0.05))),
    )


def fit_screening_woe_engine(
    train: pd.DataFrame,
    features: list[str],
    target_col: str,
    *,
    woe_engine: str = "monotone",
    woe_fit_query: str | None = None,
    woe_params: Mapping[str, Any] | None = None,
    monotone_woe_params: Mapping[str, Any] | None = None,
    categorical_features: list[str] | None = None,
) -> Any:
    """Fit a WOE engine on INS for screening steps that reuse bin boundaries."""
    from Modeling_Tool.Pipeline._common import apply_woe_fit_query

    fit_ins, _ = apply_woe_fit_query(train, woe_fit_query, target=target_col)
    if woe_engine.lower() == "monotone":
        from Modeling_Tool import MonotoneWOEBinner

        params = dict(monotone_woe_params or {})
        init_params = {k: v for k, v in params.items() if k in _MONOTONE_INIT_KEYS}
        fit_params = {k: v for k, v in params.items() if k in _MONOTONE_FIT_KEYS}
        from Modeling_Tool.Pipeline._common import as_list

        categorical = [col for col in as_list(categorical_features) if col in features]
        numeric = [col for col in features if col not in set(categorical)]
        binner = MonotoneWOEBinner(
            feature_cols=numeric,
            target_col=target_col,
            cate_feats=categorical,
            **init_params,
        )
        binner.fit(fit_ins, **fit_params)
        return binner

    from Modeling_Tool import WOE_Master

    master_params = dict(woe_params or {})
    woe_suffix = master_params.pop("woe_suffix", "_woe")
    missing_ref_value = master_params.pop("missing_ref_value", SMF_MISSING_BIN)
    master = WOE_Master(
        train_data=fit_ins,
        varlist=features,
        dep=target_col,
        graph_save_dir=None,
        woe_suffix=woe_suffix,
        missing_ref_value=missing_ref_value,
    )
    master.fit(**master_params)
    return master


def _needs_woe_bins(config: FeatureScreenConfig) -> bool:
    return config.psi_use_woe_bins or config.iv_use_woe_bins or config.corr_use_woe_bins


def _resolve_screening_binner(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    config: FeatureScreenConfig,
    prefit_woe_engine: Any | None,
) -> Any | None:
    if prefit_woe_engine is not None:
        return prefit_woe_engine
    if not _needs_woe_bins(config):
        return None
    return fit_screening_woe_engine(
        splits["ins"],
        feature_cols,
        target_col,
        woe_engine=config.woe_engine,
        woe_fit_query=config.woe_fit_query,
        woe_params=config.woe_params,
        monotone_woe_params=config.monotone_woe_params,
        categorical_features=config.categorical_features,
    )


def _woe_bins_unweighted_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    config: FeatureScreenConfig,
    *,
    prefit_woe_engine: Any | None = None,
) -> FeatureScreenResult:
    from Modeling_Tool import CorrelationFilter, PSICalculator, VarExtractionInsights

    ins, oos, oot = splits["ins"], splits["oos"], splits["oot"]
    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, None)]
    current, missing_rate_table, missing_rate_dropped = _apply_missing_rate_stage(
        ins,
        current,
        summary_rows,
        missing_rate_threshold=config.missing_rate_threshold,
        missing_rate_ref=config.missing_rate_ref,
        on_empty_stage=config.on_empty_stage,
    )
    binner = _resolve_screening_binner(splits, current, target_col, config, prefit_woe_engine)

    psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])
    if config.psi_enabled:
        psi_calc = (
            PSICalculator(buckets=config.psi_buckets, binning_engine=binner)
            if config.psi_use_woe_bins and binner is not None
            else PSICalculator(buckets=config.psi_buckets)
        )
        psi_frames = []
        if "oos" in config.psi_compare_splits and len(oos) > 0:
            psi_oos = psi_calc.calculate(expected_df=ins, current_data=oos, varlist=current)
            psi_oos = psi_oos.rename(columns={"psi": "psi_ins_oos"})[["var", "psi_ins_oos"]]
            psi_frames.append(psi_oos)
        if "oot" in config.psi_compare_splits and len(oot) > 0:
            psi_oot = psi_calc.calculate(expected_df=ins, current_data=oot, varlist=current)
            psi_oot = psi_oot.rename(columns={"psi": "psi_ins_oot"})[["var", "psi_ins_oot"]]
            psi_frames.append(psi_oot)

        if psi_frames:
            psi_table = psi_frames[0]
            for frame in psi_frames[1:]:
                psi_table = psi_table.merge(frame, on="var", how="outer")
            for col in ("psi_ins_oos", "psi_ins_oot"):
                if col not in psi_table.columns:
                    psi_table[col] = np.nan
            compare_cols = [c for c in ("psi_ins_oos", "psi_ins_oot") if c in psi_table.columns]
            psi_table["psi_max"] = psi_table[compare_cols].max(axis=1, skipna=True)
            keep = psi_table.loc[psi_table["psi_max"] < config.psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=config.on_empty_stage, threshold=config.psi_threshold,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), config.psi_threshold, None))

    iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])
    if config.iv_enabled:
        use_binner = config.iv_use_woe_bins and binner is not None
        vi = VarExtractionInsights(
            data=ins,
            dep=target_col,
            plot_path=config.plot_path or "",
            nbins=config.iv_bins,
            equal_freq=config.iv_equal_freq,
            min_bin_prop=config.iv_min_bin_prop,
            woe_binner=binner if use_binner else None,
            woe_engine="monotone" if use_binner else "master",
        )
        iv = vi.get_var_analysis_report(data=ins, varlist=current, dep=target_col, iv_cut=0.0)
        if config.plot_path and config.plot_outputs and current:
            Path(config.plot_path, "overall").mkdir(parents=True, exist_ok=True)
            plot_data = ins.copy()
            plot_data["_smf_plot_group"] = "overall"
            vi.plot_woe(
                data=plot_data,
                varlist=current,
                plot_group="_smf_plot_group",
                plot_dirname="overall",
                plot_path=config.plot_path,
            )
        if iv is not None and not iv.empty:
            iv_table = iv.rename(columns={"iv": "iv_weighted"})[["var", "iv_weighted", "n_bins", "missing_rate"]]
            keep = iv.loc[iv["iv"] >= config.iv_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "iv", summary_rows,
                on_empty_stage=config.on_empty_stage, threshold=config.iv_threshold,
            )
            summary_rows.append(_summary_row("iv", n_before, len(current), config.iv_threshold, None))

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if config.corr_enabled and len(current) > 1:
        use_binner = config.corr_use_woe_bins and binner is not None
        cf = CorrelationFilter(
            data=ins[current + [target_col]],
            dep=target_col,
            corr_cutpoint=config.corr_threshold,
            woe_binner=binner if use_binner else None,
            woe_engine="monotone" if use_binner else "master",
        )
        n_before = len(current)
        current = cf.remove_highly_correlated(current, max_iterations=config.corr_max_iterations)
        summary_rows.append(_summary_row("corr", n_before, len(current), config.corr_threshold, None))

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, None))
    return FeatureScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
    )


def _bins_to_numpy(bins: Any) -> np.ndarray:
    if isinstance(bins, pd.Series):
        return bins.to_numpy(dtype=object)
    return np.asarray(bins, dtype=object)


def _weighted_woe_bins_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    weight_col: str,
    config: FeatureScreenConfig,
    *,
    prefit_woe_engine: Any | None = None,
) -> FeatureScreenResult:
    from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

    ins, oos, oot = splits["ins"], splits["oos"], splits["oot"]
    w_ins = resolve_sample_weight(data=ins, weight_col=weight_col, expected_len=len(ins))
    y_ins = ins[target_col].astype(float).to_numpy()
    current = list(feature_cols)
    summary_rows = [_summary_row("initial", len(feature_cols), len(current), None, weight_col)]
    current, missing_rate_table, missing_rate_dropped = _apply_missing_rate_stage(
        ins,
        current,
        summary_rows,
        missing_rate_threshold=config.missing_rate_threshold,
        missing_rate_ref=config.missing_rate_ref,
        weight_col=weight_col,
        on_empty_stage=config.on_empty_stage,
    )

    binner = prefit_woe_engine
    if binner is None:
        binner = fit_screening_woe_engine(
            ins,
            current,
            target_col,
            woe_engine=config.woe_engine,
            woe_fit_query=config.woe_fit_query,
            woe_params=config.woe_params,
            monotone_woe_params=config.monotone_woe_params,
            categorical_features=config.categorical_features,
        )
    adapter = as_woe_engine(binner)

    # Assign fitted bins once per split. PSI and IV both consume these labels;
    # doing it here avoids a full WOE transform for every feature and stage.
    bin_frames: dict[str, pd.DataFrame] = {
        "ins": adapter.assign_bins_frame(ins, current),
    }
    if config.psi_enabled and "oos" in config.psi_compare_splits and len(oos) > 0:
        bin_frames["oos"] = adapter.assign_bins_frame(oos, current)
    if config.psi_enabled and "oot" in config.psi_compare_splits and len(oot) > 0:
        bin_frames["oot"] = adapter.assign_bins_frame(oot, current)

    psi_records: list[dict] = []
    if config.psi_enabled:
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            bins_ins = _bins_to_numpy(bin_frames["ins"][var])
            exp_dist = _weighted_bin_distribution(bins_ins, w_ins, config.content)
            row: dict[str, Any] = {"var": var, "psi_ins_oos": np.nan, "psi_ins_oot": np.nan}
            if "oos" in config.psi_compare_splits and len(oos) > 0:
                w_oos = resolve_sample_weight(data=oos, weight_col=weight_col, expected_len=len(oos))
                bins_oos = _bins_to_numpy(bin_frames["oos"][var])
                act = _weighted_bin_distribution(bins_oos, w_oos, config.content)
                row["psi_ins_oos"] = _psi_from_distributions(exp_dist, act, config.content)
            if "oot" in config.psi_compare_splits and len(oot) > 0:
                w_oot = resolve_sample_weight(data=oot, weight_col=weight_col, expected_len=len(oot))
                bins_oot = _bins_to_numpy(bin_frames["oot"][var])
                act = _weighted_bin_distribution(bins_oot, w_oot, config.content)
                row["psi_ins_oot"] = _psi_from_distributions(exp_dist, act, config.content)
            psi_records.append(row)

        psi_table = pd.DataFrame(psi_records) if psi_records else pd.DataFrame(
            columns=["var", "psi_ins_oos", "psi_ins_oot"],
        )
        if not psi_table.empty:
            compare_cols = [c for c in ("psi_ins_oos", "psi_ins_oot") if c in psi_table.columns]
            psi_table["psi_max"] = psi_table[compare_cols].max(axis=1, skipna=True)
            keep = psi_table.loc[psi_table["psi_max"] < config.psi_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "psi", summary_rows,
                on_empty_stage=config.on_empty_stage, weight_col=weight_col,
                threshold=config.psi_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("psi", n_before, len(current), config.psi_threshold, weight_col))
        else:
            psi_table["psi_max"] = pd.Series(dtype=float)
    else:
        psi_table = pd.DataFrame(columns=["var", "psi_ins_oos", "psi_ins_oot", "psi_max"])

    iv_records: list[dict] = []
    if config.iv_enabled:
        for var in current:
            if var not in ins.columns or ins[var].nunique(dropna=False) <= 1:
                continue
            bins_ins = _bins_to_numpy(bin_frames["ins"][var])
            x_series = ins[var]
            if pd.api.types.is_numeric_dtype(x_series):
                x_ins = x_series.to_numpy(dtype=float)
            else:
                # Categorical/object feature: IV itself is computed from the
                # already-assigned bins; ``x`` only feeds np.isfinite() inside
                # _weighted_iv_from_assigned_bins to derive the weighted
                # missing rate. A hard float cast crashes on string levels
                # (e.g. '4.高中'), so encode observed -> 1.0 / missing -> NaN,
                # matching the notna semantics of _missing_rate_for_series.
                x_ins = np.where(x_series.notna().to_numpy(), 1.0, np.nan)
            iv_val, n_b, miss = _weighted_iv_from_assigned_bins(y_ins, w_ins, bins_ins, x_ins)
            iv_records.append({
                "var": var,
                "iv_weighted": iv_val,
                "n_bins": n_b,
                "missing_rate": miss,
            })
        iv_table = pd.DataFrame(iv_records) if iv_records else pd.DataFrame(
            columns=["var", "iv_weighted", "n_bins", "missing_rate"],
        )
        if not iv_table.empty:
            keep = iv_table.loc[iv_table["iv_weighted"] >= config.iv_threshold, "var"].tolist()
            n_before = len(current)
            current = _apply_stage_keep(
                current, keep, "iv", summary_rows,
                on_empty_stage=config.on_empty_stage, weight_col=weight_col,
                threshold=config.iv_threshold, intersect=True,
            )
            summary_rows.append(_summary_row("iv", n_before, len(current), config.iv_threshold, weight_col))
    else:
        iv_table = pd.DataFrame(columns=["var", "iv_weighted", "n_bins", "missing_rate"])

    corr_dropped = pd.DataFrame(columns=["var_a", "var_b", "corr", "iv_a", "iv_b", "kept", "dropped"])
    if config.corr_enabled and len(current) > 1:
        corr = _weighted_corr_for_screen(
            ins, current, w_ins,
            corr_use_woe_bins=config.corr_use_woe_bins,
            corr_nan_policy=config.corr_nan_policy,
            corr_block_size=config.corr_block_size,
            adapter=adapter,
            binner=binner,
        )
        iv_map = dict(zip(iv_table["var"], iv_table["iv_weighted"])) if not iv_table.empty else {}
        n_before = len(current)
        current, corr_dropped = _corr_dedup_weighted(
            current,
            corr,
            iv_map,
            config.corr_threshold,
            config.corr_max_iterations,
        )
        summary_rows.append(_summary_row("corr", n_before, len(current), config.corr_threshold, weight_col))

    summary_rows.append(_summary_row("final", len(feature_cols), len(current), None, weight_col))
    return FeatureScreenResult(
        selected_features=list(current),
        iv_table=iv_table,
        psi_table=psi_table,
        corr_dropped=corr_dropped,
        summary=pd.DataFrame(summary_rows),
        missing_rate_table=missing_rate_table,
        missing_rate_dropped=missing_rate_dropped,
    )


def feature_screen(
    splits: dict[str, pd.DataFrame],
    feature_cols: list[str],
    target_col: str,
    *,
    weight_col: str | None = None,
    config: FeatureScreenConfig | None = None,
    prefit_woe_engine: Any | None = None,
) -> FeatureScreenResult:
    """Run PSI -> IV -> correlation screening on pre-split INS/OOS/OOT frames."""
    cfg = config or FeatureScreenConfig()
    use_woe_bins = _needs_woe_bins(cfg)

    if weight_col is not None:
        if use_woe_bins or prefit_woe_engine is not None:
            return _weighted_woe_bins_screen(
                splits,
                feature_cols,
                target_col,
                weight_col,
                cfg,
                prefit_woe_engine=prefit_woe_engine,
            )
        return _weighted_screen_impl(
            splits,
            feature_cols,
            target_col,
            weight_col,
            psi_enabled=cfg.psi_enabled,
            psi_threshold=cfg.psi_threshold,
            psi_compare_splits=list(cfg.psi_compare_splits),
            iv_enabled=cfg.iv_enabled,
            iv_threshold=cfg.iv_threshold,
            iv_bins=cfg.iv_bins,
            min_bin_prop=cfg.min_bin_prop,
            corr_enabled=cfg.corr_enabled,
            corr_threshold=cfg.corr_threshold,
            corr_max_iterations=cfg.corr_max_iterations,
            content=cfg.content,
            precision=cfg.precision,
            corr_use_woe_bins=cfg.corr_use_woe_bins,
            corr_nan_policy=cfg.corr_nan_policy,
            corr_block_size=cfg.corr_block_size,
            on_empty_stage=cfg.on_empty_stage,
            prefit_woe_engine=prefit_woe_engine,
            missing_rate_threshold=cfg.missing_rate_threshold,
            missing_rate_ref=cfg.missing_rate_ref,
        )

    if use_woe_bins or prefit_woe_engine is not None:
        return _woe_bins_unweighted_screen(
            splits,
            feature_cols,
            target_col,
            cfg,
            prefit_woe_engine=prefit_woe_engine,
        )

    return _legacy_unweighted_screen(
        splits,
        feature_cols,
        target_col,
        psi_enabled=cfg.psi_enabled,
        psi_threshold=cfg.psi_threshold,
        psi_compare_splits=list(cfg.psi_compare_splits),
        iv_enabled=cfg.iv_enabled,
        iv_threshold=cfg.iv_threshold,
        iv_bins=cfg.iv_bins,
        iv_min_bin_prop=cfg.iv_min_bin_prop,
        corr_enabled=cfg.corr_enabled,
        corr_threshold=cfg.corr_threshold,
        corr_max_iterations=cfg.corr_max_iterations,
        psi_buckets=cfg.psi_buckets,
        plot_path=cfg.plot_path,
        plot_outputs=cfg.plot_outputs,
        iv_equal_freq=cfg.iv_equal_freq,
        on_empty_stage=cfg.on_empty_stage,
        missing_rate_threshold=cfg.missing_rate_threshold,
        missing_rate_ref=cfg.missing_rate_ref,
    )


def feature_screen_from_dataframe(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_col: str,
    *,
    weight_col: str | None = None,
    config: FeatureScreenConfig | None = None,
    prefit_woe_engine: Any | None = None,
) -> FeatureScreenResult:
    """Convenience wrapper that resolves INS/OOS/OOT from a tagged dataframe."""
    exclude = {target_col, split_col}
    if weight_col:
        exclude.add(weight_col)
    features = [c for c in feature_cols if c not in exclude and c in data.columns]
    splits = _resolve_splits(data, split_col)
    if weight_col is not None and weight_col not in data.columns:
        raise KeyError(f"Missing weight_col {weight_col!r}")
    return feature_screen(
        splits,
        features,
        target_col,
        weight_col=weight_col,
        config=config,
        prefit_woe_engine=prefit_woe_engine,
    )


__all__ = [
    "FeatureScreenConfig",
    "FeatureScreenResult",
    "feature_screen",
    "feature_screen_from_dataframe",
    "fit_screening_woe_engine",
    "screen_config_from_mapping",
]
