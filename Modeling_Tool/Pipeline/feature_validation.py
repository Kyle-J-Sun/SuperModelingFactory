from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._common import as_list, make_dirs, safe_to_csv


@dataclass
class FeatureValidationPipelineConfig:
    output_dir: str = "output/feature_validation"
    id_col: str = "flow_id"
    apply_time_col: str = "apply_time"
    target_cols: list[str] | None = None
    new_feature_cols: list[str] | None = None
    incumbent_feature_cols: list[str] | None = None

    sample_col: str = "sample_ind"
    oot_col: str | None = "oot_flag"
    split_config: dict[str, Any] = field(default_factory=lambda: {"test_size": 0.3, "stratify": True})
    random_state: int = 42

    time_dims: list[str] = field(default_factory=lambda: ["apply_month"])
    population_dims: list[str] = field(default_factory=list)
    group_specs: dict[str, list[str]] | list[Any] | None = None
    min_group_size: int = 100

    distribution_enabled: bool = True
    distribution_params: dict[str, Any] = field(
        default_factory=lambda: {"q": [0.05, 0.15, 0.25, 0.5, 0.75, 0.95, 0.99]}
    )

    woe_enabled: bool = True
    woe_engine: str = "monotone"
    woe_params: dict[str, Any] = field(
        default_factory=lambda: {"nbins": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )
    monotone_woe_params: dict[str, Any] = field(
        default_factory=lambda: {"n_init_bins": 20, "min_bin_size": 0.03, "min_n_bins": 2}
    )
    categorical_features: list[str] | None = None
    monotone_refine_cate_enabled: bool = False
    monotone_refine_cate_params: dict[str, Any] = field(default_factory=dict)
    monotone_refine_dtree_enabled: bool = False
    monotone_refine_dtree_params: dict[str, Any] = field(default_factory=dict)
    monotone_refine_chi2_enabled: bool = False
    monotone_refine_chi2_params: dict[str, Any] = field(default_factory=dict)
    woe_plot_groups: list[str] = field(default_factory=list)

    psi_enabled: bool = True
    psi_reference_dataset: str = "ins"
    psi_reference_data: pd.DataFrame | None = None
    psi_group_dims: list[str] = field(default_factory=lambda: ["sample", "time", "population"])
    psi_use_woe_bins: bool = True
    psi_params: dict[str, Any] = field(
        default_factory=lambda: {"buckets": 10, "equal_freq": True, "min_bin_prop": 0.05}
    )

    ivks_enabled: bool = True
    ivks_group_dims: list[str] = field(default_factory=lambda: ["global", "time", "population"])
    ivks_use_woe_bins: bool = True
    ivks_params: dict[str, Any] = field(default_factory=lambda: {"iv_cut": 0.0})

    corr_enabled: bool = True
    corr_include_incumbent: bool = True
    corr_use_woe_bins: bool = True
    corr_params: dict[str, Any] = field(
        default_factory=lambda: {
            "corr_cutpoint": 0.75,
            "method": "pearson",
            "max_iterations": 10,
            "base_metric": "iv",
        }
    )

    write_outputs: bool = True
    write_excel: bool = True


@dataclass
class FeatureValidationPipelineResult:
    splits: dict[str, pd.DataFrame]
    distribution_summary: dict[str, pd.DataFrame]
    woe_artifacts: dict[str, Any]
    psi_summary: pd.DataFrame
    psi_details: dict[str, Any]
    ivks_summary: pd.DataFrame
    corr_matrix: pd.DataFrame
    high_corr_pairs: pd.DataFrame
    correlated_detail: pd.DataFrame
    validation_summary: pd.DataFrame
    output_paths: dict[str, str] = field(default_factory=dict)
    report_path: str | None = None


class FeatureValidationPipeline:
    """Feature validation workflow for new wide-table feature feeds."""

    _MONOTONE_INIT_KEYS = {
        "n_init_bins",
        "min_bin_size",
        "min_n_bins",
        "eps",
        "missing_woe",
        "special_values",
        "bin_label_decimals",
    }
    _MONOTONE_FIT_KEYS = {"chi2_binning", "chi2_p", "chi2_init_size", "n_jobs"}

    def __init__(self, config: FeatureValidationPipelineConfig | None = None):
        self.config = config or FeatureValidationPipelineConfig()

    def run(self, data: pd.DataFrame) -> FeatureValidationPipelineResult:
        cfg = self.config
        work = data.copy()
        self._add_time_columns(work)

        new_features = self._resolve_new_features(work)
        incumbent_features = self._resolve_incumbent_features(work, new_features)
        target_cols = [col for col in as_list(cfg.target_cols) if col in work.columns]
        self._validate_input(work, new_features, incumbent_features, target_cols)

        output_dir = Path(cfg.output_dir)
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(output_dir, output_dir / "figs" / "woe")

        splits = self._split_data(work, target_cols[0] if target_cols else None)
        combined = self._combine_splits(splits)
        feature_sources = self._feature_source_frame(new_features, incumbent_features)

        distribution_summary = (
            self._run_distribution(combined, new_features)
            if cfg.distribution_enabled
            else {}
        )
        woe_artifacts = (
            self._fit_woe(splits, new_features, incumbent_features, target_cols)
            if cfg.woe_enabled and target_cols
            else {}
        )
        psi_summary, psi_details = (
            self._run_psi(combined, splits, new_features, target_cols, woe_artifacts)
            if cfg.psi_enabled
            else (pd.DataFrame(), {})
        )
        ivks_summary = (
            self._run_ivks(combined, new_features, target_cols, woe_artifacts)
            if cfg.ivks_enabled and target_cols
            else pd.DataFrame()
        )
        corr_matrix, high_corr_pairs, correlated_detail = (
            self._run_correlation(combined, new_features, incumbent_features, target_cols, woe_artifacts)
            if cfg.corr_enabled
            else (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        )
        validation_summary = self._build_validation_summary(
            data=combined,
            new_features=new_features,
            incumbent_features=incumbent_features,
            target_cols=target_cols,
            feature_sources=feature_sources,
            distribution_summary=distribution_summary,
            psi_summary=psi_summary,
            ivks_summary=ivks_summary,
            high_corr_pairs=high_corr_pairs,
            woe_artifacts=woe_artifacts,
        )

        tables = self._collect_tables(
            feature_sources,
            distribution_summary,
            woe_artifacts,
            psi_summary,
            ivks_summary,
            corr_matrix,
            high_corr_pairs,
            correlated_detail,
            validation_summary,
        )
        output_paths, report_path = self._write_outputs(tables)

        return FeatureValidationPipelineResult(
            splits=splits,
            distribution_summary=distribution_summary,
            woe_artifacts=woe_artifacts,
            psi_summary=psi_summary,
            psi_details=psi_details,
            ivks_summary=ivks_summary,
            corr_matrix=corr_matrix,
            high_corr_pairs=high_corr_pairs,
            correlated_detail=correlated_detail,
            validation_summary=validation_summary,
            output_paths=output_paths,
            report_path=report_path,
        )

    def _validate_input(
        self,
        data: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
    ) -> None:
        cfg = self.config
        missing = [col for col in [cfg.id_col, cfg.apply_time_col] if col and col not in data.columns]
        missing += [col for col in new_features + incumbent_features + target_cols if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {sorted(set(missing))}")
        if not new_features:
            raise ValueError("new_feature_cols cannot be empty")
        if cfg.woe_engine.lower() not in {"monotone", "equal_freq"}:
            raise ValueError("woe_engine must be 'monotone' or 'equal_freq'")
        if cfg.psi_reference_dataset not in {"ins", "oos", "oot", "external"}:
            raise ValueError("psi_reference_dataset must be one of ins/oos/oot/external")
        if cfg.psi_reference_dataset == "external" and cfg.psi_reference_data is None:
            raise ValueError("psi_reference_data is required when psi_reference_dataset='external'")

    def _resolve_new_features(self, data: pd.DataFrame) -> list[str]:
        cfg = self.config
        if cfg.new_feature_cols:
            return list(dict.fromkeys(cfg.new_feature_cols))
        excluded = {cfg.id_col, cfg.apply_time_col, cfg.sample_col}
        if cfg.oot_col:
            excluded.add(cfg.oot_col)
        excluded.update(as_list(cfg.target_cols))
        excluded.update(as_list(cfg.incumbent_feature_cols))
        return [col for col in data.columns if col not in excluded and pd.api.types.is_numeric_dtype(data[col])]

    def _resolve_incumbent_features(self, data: pd.DataFrame, new_features: list[str]) -> list[str]:
        if not self.config.incumbent_feature_cols:
            return []
        new_set = set(new_features)
        return [col for col in dict.fromkeys(self.config.incumbent_feature_cols) if col not in new_set]

    def _add_time_columns(self, data: pd.DataFrame) -> None:
        cfg = self.config
        if cfg.apply_time_col not in data.columns:
            return
        dt = pd.to_datetime(data[cfg.apply_time_col], errors="coerce")
        if "apply_week" in cfg.time_dims and "apply_week" not in data.columns:
            data["apply_week"] = dt.dt.to_period("W").astype(str)
        if "apply_month" in cfg.time_dims and "apply_month" not in data.columns:
            data["apply_month"] = dt.dt.to_period("M").astype(str)
        if "apply_quarter" in cfg.time_dims and "apply_quarter" not in data.columns:
            data["apply_quarter"] = dt.dt.to_period("Q").astype(str)

    def _split_data(self, data: pd.DataFrame, target_col: str | None) -> dict[str, pd.DataFrame]:
        cfg = self.config
        work = data.copy()
        if cfg.sample_col in work.columns:
            lower = work[cfg.sample_col].astype(str).str.lower()
            ins = work[lower == "ins"].copy()
            oos = work[lower == "oos"].copy()
            oot = work[lower == "oot"].copy()
            if len(ins) and len(oos):
                if not len(oot):
                    oot = oos.copy()
                return {"ins": ins, "oos": oos, "oot": oot}

        if cfg.oot_col and cfg.oot_col in work.columns:
            ins_oos = work[work[cfg.oot_col] == 0].copy()
            oot = work[work[cfg.oot_col] != 0].copy()
        else:
            ins_oos = work.copy()
            oot = pd.DataFrame(columns=work.columns)

        if len(ins_oos) == 0:
            return {"ins": ins_oos.copy(), "oos": ins_oos.copy(), "oot": oot.copy()}

        if target_col and bool(cfg.split_config.get("stratify", True)) and target_col in ins_oos.columns:
            observed = ins_oos[ins_oos[target_col].notna()].copy()
            missing = ins_oos[ins_oos[target_col].isna()].copy()
            ins, oos = self._split_frame(observed, target_col)
            if len(missing):
                ins = pd.concat([ins, missing], ignore_index=True)
        else:
            ins, oos = self._split_frame(ins_oos, None)
        if len(oot) == 0:
            oot = oos.copy()
        return {"ins": ins.copy(), "oos": oos.copy(), "oot": oot.copy()}

    def _split_frame(self, data: pd.DataFrame, target_col: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        test_size = float(cfg.split_config.get("test_size", 0.3))
        random_state = int(cfg.split_config.get("random_state", cfg.random_state))
        if target_col and data[target_col].nunique(dropna=True) > 1:
            try:
                from Modeling_Tool import SampleSplitter

                splitter = SampleSplitter(test_size=test_size, random_state=random_state, stratify=True)
                return splitter.split_df(data, target=target_col)
            except Exception:
                pass
        oos = data.sample(frac=test_size, random_state=random_state)
        ins = data.drop(index=oos.index)
        return ins.reset_index(drop=True), oos.reset_index(drop=True)

    def _combine_splits(self, splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames = []
        for name, df in splits.items():
            item = df.copy()
            item["_smf_split"] = name
            frames.append(item)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _feature_source_frame(self, new_features: list[str], incumbent_features: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"feature": col, "feature_source": "new"} for col in new_features]
            + [{"feature": col, "feature_source": "incumbent"} for col in incumbent_features]
        )

    def _group_specs(self, include_global: bool = True) -> dict[str, list[str]]:
        cfg = self.config
        if cfg.group_specs is not None:
            if isinstance(cfg.group_specs, dict):
                return {str(k): list(v) for k, v in cfg.group_specs.items()}
            specs = {}
            for idx, spec in enumerate(cfg.group_specs):
                cols = [spec] if isinstance(spec, str) else list(spec)
                specs["x".join(cols) or f"group_{idx}"] = cols
            return specs
        specs: dict[str, list[str]] = {}
        if include_global:
            specs["global"] = []
        for col in cfg.time_dims:
            specs[f"time:{col}"] = [col]
        for col in cfg.population_dims:
            specs[f"population:{col}"] = [col]
        for time_col in cfg.time_dims:
            for pop_col in cfg.population_dims:
                specs[f"time_population:{time_col}x{pop_col}"] = [time_col, pop_col]
        return specs

    def _run_distribution(self, data: pd.DataFrame, features: list[str]) -> dict[str, pd.DataFrame]:
        from Modeling_Tool.Feature.Distribution_Tool import proc_means_by_grp as _proc_means_by_grp

        q = self.config.distribution_params.get("q")
        spec_missing_value = self.config.distribution_params.get("spec_missing_value")
        numeric_features = [col for col in features if pd.api.types.is_numeric_dtype(data[col])]
        categorical_features = [col for col in features if col not in numeric_features]
        tables: dict[str, pd.DataFrame] = {}
        for name, group_cols in self._group_specs(include_global=True).items():
            valid_group_cols = [col for col in group_cols if col in data.columns]
            if numeric_features:
                table = _proc_means_by_grp(
                    data,
                    numeric_features,
                    groupby=valid_group_cols,
                    spec_missing_value=spec_missing_value,
                    q=q,
                )
                table.insert(0, "group_spec", name)
                tables[f"numeric_{name}"] = table
            if categorical_features:
                tables[f"categorical_{name}"] = self._categorical_distribution(data, categorical_features, valid_group_cols, name)
        return tables

    def _categorical_distribution(
        self,
        data: pd.DataFrame,
        features: list[str],
        group_cols: list[str],
        group_spec: str,
    ) -> pd.DataFrame:
        rows = []
        grouped = [((), data)] if not group_cols else data.groupby(group_cols, dropna=False)
        for group_value, sub in grouped:
            if not isinstance(group_value, tuple):
                group_value = (group_value,)
            group_info = dict(zip(group_cols, group_value))
            for feature in features:
                vc = sub[feature].value_counts(dropna=False)
                top = vc.head(5)
                rows.append(
                    {
                        "group_spec": group_spec,
                        **group_info,
                        "attribute": feature,
                        "N_ALL": len(sub),
                        "N": int(sub[feature].notna().sum()),
                        "MISSING_RATE": float(sub[feature].isna().mean()),
                        "N_UNIQUE": int(sub[feature].nunique(dropna=True)),
                        "TOP_VALUES": "; ".join(f"{idx}:{cnt}" for idx, cnt in top.items()),
                    }
                )
        return pd.DataFrame(rows)

    def _fit_woe(
        self,
        splits: dict[str, pd.DataFrame],
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
    ) -> dict[str, Any]:
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        cfg = self.config
        fit_features = list(new_features)
        if cfg.corr_include_incumbent and cfg.corr_use_woe_bins:
            fit_features = list(dict.fromkeys(fit_features + incumbent_features))

        by_target = {}
        woe_tables = []
        refine_rows = []
        for target in target_cols:
            train = splits["ins"][splits["ins"][target].notna()].copy()
            if len(train) < max(10, cfg.min_group_size):
                refine_rows.append({"target": target, "step": "fit", "status": "skipped_min_group_size", "n": len(train)})
                continue
            try:
                engine = (
                    self._fit_monotone_binner(train, fit_features, target, refine_rows)
                    if cfg.woe_engine.lower() == "monotone"
                    else self._fit_woe_master(train, fit_features, target)
                )
                adapter = as_woe_engine(engine)
                table = adapter.get_woe_table(fit_features)
                table.insert(0, "target", target)
                woe_tables.append(table)
                woe_splits = {
                    name: adapter.transform(df, varlist=fit_features)
                    for name, df in splits.items()
                }
                self._plot_woe(engine, adapter, train, fit_features, target)
                by_target[target] = {
                    "engine": engine,
                    "adapter": adapter,
                    "woe_splits": woe_splits,
                    "features": fit_features,
                }
            except Exception as exc:
                refine_rows.append({"target": target, "step": "fit", "status": "error", "error": repr(exc)})
        return {
            "by_target": by_target,
            "woe_table": pd.concat(woe_tables, ignore_index=True) if woe_tables else pd.DataFrame(),
            "refine_summary": pd.DataFrame(refine_rows),
        }

    def _fit_monotone_binner(
        self,
        train: pd.DataFrame,
        features: list[str],
        target: str,
        refine_rows: list[dict[str, Any]],
    ) -> Any:
        from Modeling_Tool import MonotoneWOEBinner

        cfg = self.config
        categorical = [col for col in as_list(cfg.categorical_features) if col in features]
        numeric = [col for col in features if col not in set(categorical)]
        params = dict(cfg.monotone_woe_params or {})
        init_params = {k: v for k, v in params.items() if k in self._MONOTONE_INIT_KEYS}
        fit_params = {k: v for k, v in params.items() if k in self._MONOTONE_FIT_KEYS}
        binner = MonotoneWOEBinner(
            feature_cols=numeric,
            target_col=target,
            cate_feats=categorical,
            **init_params,
        )
        binner.fit(train, **fit_params)
        refine_rows.append({"target": target, "step": "fit_monotone", "status": "ok", "features": ",".join(features)})

        if cfg.monotone_refine_cate_enabled:
            params = dict(cfg.monotone_refine_cate_params or {})
            params.setdefault("features", categorical)
            binner.refine_cate(**params)
            refine_rows.append({"target": target, "step": "refine_cate", "status": "ok", "params": repr(params)})
        if cfg.monotone_refine_dtree_enabled:
            params = dict(cfg.monotone_refine_dtree_params or {})
            params.setdefault("features", numeric)
            binner.refine_dtree(train, **params)
            refine_rows.append({"target": target, "step": "refine_dtree", "status": "ok", "params": repr(params)})
        if cfg.monotone_refine_chi2_enabled:
            params = dict(cfg.monotone_refine_chi2_params or {})
            params.setdefault("features", numeric)
            binner.refine_chi2(train, **params)
            refine_rows.append({"target": target, "step": "refine_chi2", "status": "ok", "params": repr(params)})
        return binner

    def _fit_woe_master(self, train: pd.DataFrame, features: list[str], target: str) -> Any:
        from Modeling_Tool import WOE_Master

        cfg = self.config
        graph_dir = str(Path(cfg.output_dir) / "figs" / "woe" / target)
        master = WOE_Master(
            train_data=train,
            varlist=features,
            dep=target,
            graph_save_dir=graph_dir,
            woe_suffix=cfg.woe_params.get("woe_suffix", "_woe"),
            missing_ref_value=cfg.woe_params.get("missing_ref_value", -999999),
        )
        fit_params = {k: v for k, v in cfg.woe_params.items() if k not in {"woe_suffix", "missing_ref_value"}}
        master.fit(**fit_params)
        return master

    def _plot_woe(self, engine: Any, adapter: Any, train: pd.DataFrame, features: list[str], target: str) -> None:
        cfg = self.config
        if not cfg.write_outputs:
            return
        base_dir = Path(cfg.output_dir) / "figs" / "woe" / target
        make_dirs(base_dir)
        try:
            if adapter.get_engine_name() == "monotone" and hasattr(engine, "plot_woe_graph"):
                engine.plot_woe_graph(str(base_dir))
                for group in cfg.woe_plot_groups:
                    if group in train.columns:
                        engine.plot_woe_graph(str(base_dir / f"by_{group}"), group_name=group, _df_for_group=train)
            elif hasattr(engine, "plot_bivar_graph"):
                transformed = engine.transform(train)
                engine.plot_bivar_graph(transformed, dirname=str(base_dir))
                for group in cfg.woe_plot_groups:
                    if group in transformed.columns:
                        engine.plot_bivar_graph(transformed, group=group, dirname=str(base_dir / f"by_{group}"))
        except Exception:
            return

    def _run_psi(
        self,
        combined: pd.DataFrame,
        splits: dict[str, pd.DataFrame],
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        from Modeling_Tool import PSICalculator

        cfg = self.config
        reference = cfg.psi_reference_data.copy() if cfg.psi_reference_dataset == "external" else splits[cfg.psi_reference_dataset].copy()
        if cfg.psi_reference_dataset == "external":
            self._add_time_columns(reference)
        group_cols = self._resolve_group_dim_columns(cfg.psi_group_dims)
        if "sample" in cfg.psi_group_dims:
            group_cols = ["_smf_split"] + group_cols
        group_cols = list(dict.fromkeys([col for col in group_cols if col in combined.columns]))

        rows = []
        details: dict[str, Any] = {}
        engines = self._psi_engines(target_cols, woe_artifacts)
        for target_key, engine in engines.items():
            calc = PSICalculator(binning_engine=engine, **cfg.psi_params) if cfg.psi_use_woe_bins and engine is not None else PSICalculator(**cfg.psi_params)
            for group_col in group_cols or [None]:
                try:
                    result = calc.calculate(
                        reference,
                        combined,
                        features,
                        group_by=group_col,
                        group_name=group_col,
                        return_details=True,
                    )
                    if isinstance(result, dict) and "psi" in result:
                        if "details" in result:
                            details[f"{target_key}:{group_col}"] = result["details"]
                        result = result["psi"]
                    if isinstance(result, tuple):
                        result, detail = result
                        details[f"{target_key}:{group_col}"] = detail
                    result = pd.DataFrame(result)
                    if group_col is None and "psi" not in result.columns:
                        result = pd.DataFrame({"psi": result.iloc[:, 0]})
                    result["target"] = target_key
                    result["group_col"] = group_col or "global"
                    rows.append(result)
                except Exception as exc:
                    rows.append(pd.DataFrame({"target": [target_key], "group_col": [group_col or "global"], "error": [repr(exc)]}))
        if not rows:
            return pd.DataFrame(), details
        return pd.concat(rows, ignore_index=True), details

    def _psi_engines(self, target_cols: list[str], woe_artifacts: dict[str, Any]) -> dict[str, Any]:
        by_target = woe_artifacts.get("by_target", {}) if woe_artifacts else {}
        if by_target:
            return {target: item.get("engine") for target, item in by_target.items()}
        return {"no_target": None}

    def _resolve_group_dim_columns(self, dims: list[str]) -> list[str]:
        cfg = self.config
        cols = []
        if "time" in dims:
            cols.extend(cfg.time_dims)
        if "population" in dims:
            cols.extend(cfg.population_dims)
        cols.extend([dim for dim in dims if dim not in {"global", "sample", "time", "population"}])
        return cols

    def _run_ivks(
        self,
        combined: pd.DataFrame,
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        rows = []
        for target in target_cols:
            data = combined[combined[target].notna()].copy()
            if len(data) < self.config.min_group_size:
                continue
            group_specs = self._ivks_group_specs()
            for group_name, group_cols in group_specs.items():
                valid_group_cols = [col for col in group_cols if col in data.columns]
                if not valid_group_cols:
                    rows.append(self._single_ivks(data, features, target, "global", {}, woe_artifacts))
                else:
                    for group_value, sub in data.groupby(valid_group_cols, dropna=False):
                        if len(sub) < self.config.min_group_size:
                            continue
                        if not isinstance(group_value, tuple):
                            group_value = (group_value,)
                        group_info = dict(zip(valid_group_cols, group_value))
                        rows.append(self._single_ivks(sub, features, target, group_name, group_info, woe_artifacts))
        frames = [df for df in rows if isinstance(df, pd.DataFrame) and not df.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _ivks_group_specs(self) -> dict[str, list[str]]:
        dims = set(self.config.ivks_group_dims)
        specs = {}
        if "global" in dims:
            specs["global"] = []
        if "time" in dims:
            specs.update({f"time:{col}": [col] for col in self.config.time_dims})
        if "population" in dims:
            specs.update({f"population:{col}": [col] for col in self.config.population_dims})
        for dim in dims - {"global", "time", "population"}:
            specs[str(dim)] = [str(dim)]
        return specs

    def _single_ivks(
        self,
        data: pd.DataFrame,
        features: list[str],
        target: str,
        group_spec: str,
        group_info: dict[str, Any],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        from Modeling_Tool import VarExtractionInsights

        cfg = self.config
        binner = None
        if cfg.ivks_use_woe_bins:
            binner = woe_artifacts.get("by_target", {}).get(target, {}).get("engine")
        if binner is not None:
            report = self._ivks_from_binner(data, features, target, binner)
            if report.empty:
                return report
            report.insert(0, "target", target)
            report.insert(1, "group_spec", group_spec)
            for key, value in group_info.items():
                report[key] = value
            return report
        params = dict(cfg.ivks_params or {})
        iv_cut = float(params.pop("iv_cut", 0.0))
        insights = VarExtractionInsights(
            data=data,
            dep=target,
            plot_path=None,
            woe_binner=binner,
            woe_engine="monotone" if binner is not None else "master",
            **params,
        )
        report = insights.get_var_analysis_report(data, features, dep=target, iv_cut=iv_cut)
        if report is None or len(report) == 0:
            return pd.DataFrame()
        report = report.copy()
        report.insert(0, "target", target)
        report.insert(1, "group_spec", group_spec)
        for key, value in group_info.items():
            report[key] = value
        return report

    def _ivks_from_binner(self, data: pd.DataFrame, features: list[str], target: str, binner: Any) -> pd.DataFrame:
        from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine

        adapter = as_woe_engine(binner)
        if adapter is None:
            return pd.DataFrame()
        target_series = data[target].astype(int)
        total_bad = max(float((target_series == 1).sum()), 1.0)
        total_good = max(float((target_series == 0).sum()), 1.0)
        overall_bad = max(float(target_series.mean()), 1e-6)
        rows = []
        for var in features:
            if var not in data.columns or data[var].nunique(dropna=False) <= 1:
                continue
            try:
                bins = adapter.assign_bins(data, var)
                tmp = pd.DataFrame({"bin": bins, target: target_series})
                grouped = tmp.groupby("bin", dropna=False)[target].agg(["count", "sum"]).reset_index()
                grouped = grouped.rename(columns={"count": "n", "sum": "n_bad"})
                grouped["n_good"] = grouped["n"] - grouped["n_bad"]
                grouped["bad_pct"] = grouped["n_bad"] / total_bad
                grouped["good_pct"] = grouped["n_good"] / total_good
                grouped["woe"] = np.log((grouped["bad_pct"] + 1e-6) / (grouped["good_pct"] + 1e-6))
                grouped["iv_component"] = (grouped["bad_pct"] - grouped["good_pct"]) * grouped["woe"]
                grouped["bad_rate"] = grouped["n_bad"] / grouped["n"].replace(0, np.nan)
                ordered = grouped.sort_values("bad_rate", ascending=False).reset_index(drop=True)
                ks = float((ordered["bad_pct"].cumsum() - ordered["good_pct"].cumsum()).abs().max())
                lift = float((ordered["bad_rate"] / overall_bad).replace([np.inf, -np.inf], np.nan).max())
                series = data[var]
                is_numeric = pd.api.types.is_numeric_dtype(series)
                rows.append(
                    {
                        "var": var,
                        "n_all": len(series),
                        "n": int(series.notna().sum()),
                        "ks_in_gains": ks,
                        "lift_in_gains": lift,
                        "iv": float(grouped["iv_component"].sum()),
                        "n_bump": int(grouped.shape[0]),
                        "missing_rate": float(series.isna().mean()),
                        "min": float(series.min()) if is_numeric else np.nan,
                        "mean": float(series.mean()) if is_numeric else np.nan,
                        "max": float(series.max()) if is_numeric else np.nan,
                        "n_bins": int(grouped.shape[0]),
                    }
                )
            except Exception:
                continue
        if not rows:
            return pd.DataFrame()
        report = pd.DataFrame(rows).round(4)
        iv_cut = float((self.config.ivks_params or {}).get("iv_cut", 0.0))
        return report[report["iv"] >= iv_cut].reset_index(drop=True)

    def _run_correlation(
        self,
        combined: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cfg = self.config
        features = list(new_features)
        if cfg.corr_include_incumbent:
            features = list(dict.fromkeys(features + incumbent_features))
        numeric_features = [col for col in features if col in combined.columns and pd.api.types.is_numeric_dtype(combined[col])]
        method = str(cfg.corr_params.get("method", "pearson"))
        threshold = float(cfg.corr_params.get("corr_cutpoint", 0.75))
        corr_matrix = combined[numeric_features].corr(method=method) if len(numeric_features) >= 2 else pd.DataFrame()
        high_corr_pairs = self._high_corr_pairs(corr_matrix, threshold, new_features, incumbent_features)
        correlated_detail = self._correlated_detail(combined, numeric_features, target_cols, woe_artifacts)
        return corr_matrix, high_corr_pairs, correlated_detail

    def _high_corr_pairs(
        self,
        corr_matrix: pd.DataFrame,
        threshold: float,
        new_features: list[str],
        incumbent_features: list[str],
    ) -> pd.DataFrame:
        if corr_matrix.empty:
            return pd.DataFrame(columns=["var1", "var2", "corr", "abs_corr", "pair_type"])
        rows = []
        new_set = set(new_features)
        incumbent_set = set(incumbent_features)
        cols = list(corr_matrix.columns)
        for i, var1 in enumerate(cols):
            for var2 in cols[i + 1:]:
                corr = corr_matrix.loc[var1, var2]
                if pd.notna(corr) and abs(corr) > threshold:
                    if var1 in new_set and var2 in new_set:
                        pair_type = "new_new"
                    elif (var1 in new_set and var2 in incumbent_set) or (var2 in new_set and var1 in incumbent_set):
                        pair_type = "new_incumbent"
                    else:
                        pair_type = "incumbent_incumbent"
                    rows.append({"var1": var1, "var2": var2, "corr": corr, "abs_corr": abs(corr), "pair_type": pair_type})
        return pd.DataFrame(rows)

    def _correlated_detail(
        self,
        combined: pd.DataFrame,
        features: list[str],
        target_cols: list[str],
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        if not target_cols or len(features) < 2:
            return pd.DataFrame()
        from Modeling_Tool import CorrelationFilter

        cfg = self.config
        rows = []
        for target in target_cols:
            data = combined[combined[target].notna()].copy()
            if len(data) < cfg.min_group_size or data[target].nunique() < 2:
                continue
            params = dict(cfg.corr_params or {})
            max_iterations = int(params.pop("max_iterations", 10))
            binner = woe_artifacts.get("by_target", {}).get(target, {}).get("engine") if cfg.corr_use_woe_bins else None
            filt = CorrelationFilter(data=data, dep=target, woe_binner=binner, **params)
            keep = filt.remove_highly_correlated(features, max_iterations=max_iterations)
            removed = [feature for feature in features if feature not in keep]
            for anchor, payload in getattr(filt, "correlated_dict", {}).items():
                corr = payload.get("corr", pd.DataFrame()).copy()
                gains = payload.get("gains", pd.DataFrame()).copy()
                metric_cols = [col for col in ["var", "iv", "ks_in_gains", "lift_in_gains"] if col in gains.columns]
                metric_map = gains[metric_cols].set_index("var").to_dict("index") if "var" in gains.columns else {}
                for _, row in corr.iterrows():
                    for var in [row.get("VAR1"), row.get("VAR2")]:
                        metrics = metric_map.get(var, {})
                        rows.append(
                            {
                                "target": target,
                                "anchor_var": anchor,
                                "var": var,
                                "corr_var1": row.get("VAR1"),
                                "corr_var2": row.get("VAR2"),
                                "corr": row.get("CORR"),
                                "recommended_action": "keep" if var in keep else "remove",
                                "iv": metrics.get("iv"),
                                "ks_in_gains": metrics.get("ks_in_gains"),
                                "lift_in_gains": metrics.get("lift_in_gains"),
                            }
                        )
            if not rows and removed:
                rows.append({"target": target, "recommended_action": "remove", "var": ",".join(removed)})
        return pd.DataFrame(rows)

    def _build_validation_summary(
        self,
        data: pd.DataFrame,
        new_features: list[str],
        incumbent_features: list[str],
        target_cols: list[str],
        feature_sources: pd.DataFrame,
        distribution_summary: dict[str, pd.DataFrame],
        psi_summary: pd.DataFrame,
        ivks_summary: pd.DataFrame,
        high_corr_pairs: pd.DataFrame,
        woe_artifacts: dict[str, Any],
    ) -> pd.DataFrame:
        rows = [
            {"metric": "n_rows", "value": len(data)},
            {"metric": "n_new_features", "value": len(new_features)},
            {"metric": "n_incumbent_features", "value": len(incumbent_features)},
            {"metric": "n_targets", "value": len(target_cols)},
            {"metric": "woe_engine", "value": self.config.woe_engine},
            {"metric": "distribution_tables", "value": len(distribution_summary)},
            {"metric": "psi_rows", "value": len(psi_summary)},
            {"metric": "ivks_rows", "value": len(ivks_summary)},
            {"metric": "high_corr_pairs", "value": len(high_corr_pairs)},
            {"metric": "woe_targets", "value": len(woe_artifacts.get("by_target", {})) if woe_artifacts else 0},
            {"metric": "feature_sources", "value": feature_sources["feature_source"].value_counts().to_dict()},
        ]
        return pd.DataFrame(rows)

    def _collect_tables(
        self,
        feature_sources: pd.DataFrame,
        distribution_summary: dict[str, pd.DataFrame],
        woe_artifacts: dict[str, Any],
        psi_summary: pd.DataFrame,
        ivks_summary: pd.DataFrame,
        corr_matrix: pd.DataFrame,
        high_corr_pairs: pd.DataFrame,
        correlated_detail: pd.DataFrame,
        validation_summary: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        tables = {
            "validation_summary": validation_summary,
            "feature_sources": feature_sources,
            "woe_table": woe_artifacts.get("woe_table", pd.DataFrame()) if woe_artifacts else pd.DataFrame(),
            "woe_refine_summary": woe_artifacts.get("refine_summary", pd.DataFrame()) if woe_artifacts else pd.DataFrame(),
            "psi_summary": psi_summary,
            "ivks_summary": ivks_summary,
            "corr_matrix": corr_matrix.reset_index(names="feature") if not corr_matrix.empty else corr_matrix,
            "high_corr_pairs": high_corr_pairs,
            "correlated_detail": correlated_detail,
        }
        for name, df in distribution_summary.items():
            tables[f"distribution_{name}"] = df
        return tables

    def _write_outputs(self, tables: dict[str, pd.DataFrame]) -> tuple[dict[str, str], str | None]:
        cfg = self.config
        output_paths: dict[str, str] = {}
        report_path = None
        if not (cfg.write_outputs or cfg.write_excel):
            return output_paths, report_path
        output_dir = Path(cfg.output_dir)
        make_dirs(output_dir)
        if cfg.write_outputs:
            for name, df in tables.items():
                path = output_dir / f"{name}.csv"
                safe_to_csv(df, path, index=False)
                output_paths[name] = str(path.resolve())
        if cfg.write_excel:
            excel_path = output_dir / "Feature_Validation_Report.xlsx"
            self._write_excel_report(excel_path, tables)
            report_path = str(excel_path.resolve())
            output_paths["excel_report"] = report_path
        return output_paths, report_path

    def _write_excel_report(self, excel_path: Path, tables: dict[str, pd.DataFrame]) -> None:
        from ExcelMaster.ExcelMaster import ExcelMaster

        em = ExcelMaster(str(excel_path), verbose=False)
        used_names: set[str] = set()
        for name, df in tables.items():
            if not isinstance(df, pd.DataFrame) or df.empty:
                continue
            sheet_name = self._excel_sheet_name(name, used_names)
            ws = em.add_worksheet(sheet_name, zoom_perc=90)
            em.write_dataframe(ws, df=df, title=name, index=False)
        if not used_names:
            ws = em.add_worksheet("Summary", zoom_perc=90)
            em.write_dataframe(ws, df=pd.DataFrame({"message": ["No non-empty tables generated."]}), title="Summary", index=False)
        em.close_workbook()

    @staticmethod
    def _excel_sheet_name(name: str, used_names: set[str]) -> str:
        invalid = set("[]:*?/\\")
        base = "".join(ch if ch not in invalid else "_" for ch in name)[:31] or "Sheet"
        candidate = base
        idx = 1
        while candidate in used_names:
            suffix = f"_{idx}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            idx += 1
        used_names.add(candidate)
        return candidate
