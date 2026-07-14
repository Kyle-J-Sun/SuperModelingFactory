"""WOE-engine-aware wrappers for feature screening tools.

The original Feature modules are compiled in release builds, so mutating their
class objects at import time is fragile.  These wrappers delegate to the original
classes for legacy behavior and only take over when a fitted WOE engine is
explicitly supplied.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from Modeling_Tool._utils.robust import smf_logger
from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine
from .PSI_Tool import PSICalculator as _BasePSICalculator
from .PSI_Tool import _psi_distributions_from_counts, _validate_psi_bucket_policy
from .Feature_Insights import (
    CorrelationFilter as _BaseCorrelationFilter,
    VarExtractionInsights as _BaseVarExtractionInsights,
    var_corr_filter,
)
from .Distribution_Tool import proc_means_for_screening


def _psi_from_bins(
    expected_bins: pd.Series,
    current_bins: pd.Series,
    content: float,
    psi_missing_bucket_policy: str,
) -> float:
    expected_count = expected_bins.value_counts(normalize=False, dropna=False)
    current_count = current_bins.value_counts(normalize=False, dropna=False)
    _, _, psi_values, _ = _psi_distributions_from_counts(
        expected_count,
        current_count,
        float(len(expected_bins)),
        float(len(current_bins)),
        content=content,
        policy=psi_missing_bucket_policy,
    )
    return float(psi_values.sum())


def _make_monotone_adapter(owner, data: pd.DataFrame, varlist: list[str]):
    if getattr(owner, "woe_binner", None) is not None:
        return as_woe_engine(owner.woe_binner)
    if getattr(owner, "woe_engine", "master") != "monotone":
        return None

    from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

    params = dict(getattr(owner, "woe_engine_params", {}) or {})
    fit_params = params.pop("fit_params", {}) if "fit_params" in params else {}
    binner = MonotoneWOEBinner(
        feature_cols=varlist,
        target_col=owner.dep,
        special_values=owner.spec_values,
        **params,
    )
    binner.fit(
        data,
        chi2_binning=owner.chi2_method,
        chi2_p=owner.chi2_p,
        chi2_init_size=owner.init_equi_bins,
        **fit_params,
    )
    owner.woe_binner = binner
    return as_woe_engine(binner)


def _screening_summary_from_engine(
    data: pd.DataFrame,
    varlist: list[str],
    dep: str,
    adapter,
    iv_cut: float,
    missing_rate_ref,
    failed_variables: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows = []
    target = data[dep]
    total_bad = max(float((target == 1).sum()), 1.0)
    total_good = max(float((target == 0).sum()), 1.0)
    valid_vars = [
        var
        for var in varlist
        if var in data.columns and data[var].nunique(dropna=False) > 1
    ]
    assigned_bins = None
    assign_bins_frame = getattr(adapter, "assign_bins_frame", None)
    if callable(assign_bins_frame):
        try:
            assigned_bins = assign_bins_frame(data, valid_vars)
        except (TypeError, ValueError, KeyError, AttributeError, np.linalg.LinAlgError):
            # Keep variable-level failure isolation for custom and legacy adapters.
            # A production adapter normally succeeds here and uses the bulk path.
            assigned_bins = None

    for var in tqdm(valid_vars):
        try:
            bins = assigned_bins[var] if assigned_bins is not None else adapter.assign_bins(data, var)
            tmp = pd.DataFrame({"bin": bins, dep: target})
            grouped = tmp.groupby("bin", dropna=False)[dep].agg(["count", "sum"]).reset_index()
            grouped = grouped.rename(columns={"count": "n", "sum": "n_bad"})
            grouped["n_good"] = grouped["n"] - grouped["n_bad"]
            grouped["bad_pct"] = grouped["n_bad"] / total_bad
            grouped["good_pct"] = grouped["n_good"] / total_good
            grouped["woe"] = np.log((grouped["bad_pct"] + 1e-6) / (grouped["good_pct"] + 1e-6))
            grouped["iv"] = (grouped["bad_pct"] - grouped["good_pct"]) * grouped["woe"]
            grouped["avg_bad"] = grouped["n_bad"] / grouped["n"].replace(0, np.nan)
            grouped = grouped.sort_values("avg_bad", ascending=False).reset_index(drop=True)
            ks = (grouped["bad_pct"].cumsum() - grouped["good_pct"].cumsum()).abs().max()
            overall_bad = max(float((target == 1).mean()), 1e-6)
            lift = float((grouped["avg_bad"] / overall_bad).replace([np.inf, -np.inf], np.nan).max())
            rows.append({
                "var": var,
                "ks_in_gains": float(ks),
                "lift_in_gains": lift,
                "iv": float(grouped["iv"].sum()),
                "n_bump": int(grouped.shape[0]),
                "n_bins": int(grouped.shape[0]),
            })
        except (TypeError, ValueError, KeyError, ZeroDivisionError, np.linalg.LinAlgError) as exc:
            row = smf_logger.record_and_continue(var, exc, stage="woe_engine_feature_patch")
            if failed_variables is not None:
                failed_variables.append((row["feature"], row["exception_type"]))
            continue

    columns = [
        "var", "n_all", "n", "ks_in_gains", "lift_in_gains", "iv",
        "n_bump", "missing_rate", "min", "mean", "max", "n_bins",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    high_iv_summary = pd.DataFrame(rows).query(f"iv >= {iv_cut}").round(4)
    high_iv_varlist = high_iv_summary["var"].tolist()
    if high_iv_varlist:
        means = proc_means_for_screening(data, high_iv_varlist, spec_missing_value=missing_rate_ref)
        summary = high_iv_summary.merge(
            means,
            left_on="var",
            right_on="attribute",
            how="left",
        )
    else:
        summary = high_iv_summary.copy()
        for col in ["N_ALL", "N", "MISSING_RATE", "MIN", "MEAN", "MAX"]:
            summary[col] = np.nan

    summary.columns = [str(c).lower() for c in summary.columns]
    return summary[columns]


class PSICalculator:
    def __init__(
        self,
        buckets: int = 10,
        equal_freq: bool = True,
        min_bin_prop: float = 0.05,
        content: float = 1e-6,
        precision: int = 5,
        binning_engine=None,
        missing_policy: str = "include",
        psi_missing_bucket_policy: str = "smooth_laplace",
        feature_block_size: int | None = 64,
    ):
        _validate_psi_bucket_policy(psi_missing_bucket_policy, "Feature.PSICalculator.__init__")
        self._base = _BasePSICalculator(
            buckets=buckets,
            equal_freq=equal_freq,
            min_bin_prop=min_bin_prop,
            content=content,
            precision=precision,
            missing_policy=missing_policy,
            psi_missing_bucket_policy=psi_missing_bucket_policy,
            feature_block_size=feature_block_size,
        )
        self.buckets = buckets
        self.equal_freq = equal_freq
        self.min_bin_prop = min_bin_prop
        self.content = content
        self.precision = precision
        self.missing_policy = missing_policy
        self.psi_missing_bucket_policy = psi_missing_bucket_policy
        if feature_block_size is not None and int(feature_block_size) <= 0:
            raise ValueError("feature_block_size must be a positive integer or None")
        self.feature_block_size = feature_block_size
        self.binning_engine = binning_engine
        self._woe_engine_adapter = as_woe_engine(binning_engine) if binning_engine is not None else None

    def __getattr__(self, name):
        return getattr(self._base, name)

    def _prepare_woe_bins(self, expected_df, current_data, varlist):
        adapter = self._woe_engine_adapter
        if adapter is None:
            return None
        return (
            adapter.assign_bins_frame(
                expected_df,
                varlist,
                feature_block_size=self.feature_block_size,
            ),
            adapter.assign_bins_frame(
                current_data,
                varlist,
                feature_block_size=self.feature_block_size,
            ),
        )

    @staticmethod
    def _group_codes(current_data, resolved_group):
        if resolved_group is None:
            return np.zeros(len(current_data), dtype=np.int64), [(None, {})]

        group_cols = [resolved_group] if isinstance(resolved_group, str) else list(resolved_group)
        group_frame = current_data[group_cols].copy()
        for col in group_cols:
            group_frame[col] = group_frame[col].where(group_frame[col].notna(), "__NULL__")
        group_key = group_cols[0] if len(group_cols) == 1 else group_cols
        grouped = group_frame.groupby(group_key, dropna=False, sort=True)
        codes = grouped.ngroup().to_numpy(dtype=np.int64)
        keys = list(grouped.size().index)
        groups = []
        for key in keys:
            values = key if isinstance(key, tuple) else (key,)
            groups.append((key, dict(zip(group_cols, values))))
        return codes, groups

    def _calculate_prebinned(
        self,
        expected_bins,
        current_bins,
        current_data,
        varlist,
        resolved_group,
        return_details,
        bucket_policy,
    ):
        group_codes, groups = self._group_codes(current_data, resolved_group)
        n_groups = len(groups)
        rows = []
        detail = {}

        for var in varlist:
            expected_values = expected_bins[var].to_numpy(dtype=object)
            current_values = current_bins[var].to_numpy(dtype=object)
            joined = np.concatenate([expected_values, current_values])
            bin_codes, categories = pd.factorize(joined, sort=False)
            expected_codes = bin_codes[: len(expected_values)]
            current_codes = bin_codes[len(expected_values) :]
            n_bins = len(categories)
            expected_count_values = np.bincount(expected_codes, minlength=n_bins)
            current_count_values = np.bincount(
                group_codes * n_bins + current_codes,
                minlength=n_groups * n_bins,
            ).reshape(n_groups, n_bins)
            expected_count = pd.Series(expected_count_values, index=categories)

            for group_idx, (group_value, group_info) in enumerate(groups):
                current_count = pd.Series(current_count_values[group_idx], index=categories)
                _, _, psi_values, _ = _psi_distributions_from_counts(
                    expected_count,
                    current_count,
                    float(expected_count_values.sum()),
                    float(current_count_values[group_idx].sum()),
                    content=self.content,
                    policy=bucket_policy,
                )
                row = {"var": var, "psi": round(float(psi_values.sum()), self.precision)}
                row.update(group_info)
                rows.append(row)
                if return_details:
                    detail_key = var if resolved_group is None else (var, group_value)
                    exp_total = max(float(expected_count_values.sum()), 1.0)
                    cur_total = max(float(current_count_values[group_idx].sum()), 1.0)
                    detail[detail_key] = {
                        "expected_bins": (expected_count / exp_total).sort_values(ascending=False),
                        "current_bins": (current_count / cur_total).sort_values(ascending=False),
                    }
        result = pd.DataFrame(rows)
        return (result, detail) if return_details else result

    def calculate(
        self,
        expected_df,
        current_data,
        varlist,
        group_by=None,
        group_name=None,
        return_details=False,
        missing_policy=None,
        psi_missing_bucket_policy=None,
    ):
        adapter = self._woe_engine_adapter
        effective_missing_policy = missing_policy if missing_policy is not None else self.missing_policy
        effective_bucket_policy = (
            psi_missing_bucket_policy
            if psi_missing_bucket_policy is not None
            else self.psi_missing_bucket_policy
        )
        if adapter is None:
            return self._base.calculate(
                expected_df,
                current_data,
                varlist,
                group_by,
                group_name,
                return_details,
                missing_policy=effective_missing_policy,
                psi_missing_bucket_policy=effective_bucket_policy,
            )

        resolved_group = group_name if group_name is not None else group_by
        expected_bins, current_bins = self._prepare_woe_bins(expected_df, current_data, varlist)
        return self._calculate_prebinned(
            expected_bins,
            current_bins,
            current_data,
            varlist,
            resolved_group,
            return_details,
            effective_bucket_policy,
        )


class VarExtractionInsights:
    def __init__(
        self,
        data,
        dep,
        plot_path,
        nbins=10,
        equal_freq=True,
        min_bin_prop=0.05,
        precision=5,
        chi2_method=False,
        chi2_p=0.9,
        init_equi_bins=5000,
        tree_binning=True,
        include_missing=True,
        seed=3407,
        missing_rate_ref=-999999,
        spec_values=None,
        woe_engine="master",
        woe_binner=None,
        woe_engine_params=None,
    ):
        self._base = _BaseVarExtractionInsights(
            data, dep, plot_path, nbins, equal_freq, min_bin_prop, precision,
            chi2_method, chi2_p, init_equi_bins, tree_binning, include_missing,
            seed, missing_rate_ref, spec_values,
        )
        self.data = data
        self.dep = dep
        self.plot_path = plot_path
        self.nbins = nbins
        self.equal_freq = equal_freq
        self.min_bin_prop = min_bin_prop
        self.precision = precision
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.tree_binning = tree_binning
        self.include_missing = include_missing
        self.seed = seed
        self.missing_rate_ref = missing_rate_ref
        self.spec_values = spec_values if spec_values is not None else []
        self.woe_engine = woe_engine
        self.woe_binner = woe_binner
        self.woe_engine_params = woe_engine_params or {}
        self.failed_variables = []

    def __getattr__(self, name):
        return getattr(self._base, name)

    @staticmethod
    def remove_folder(file_path):
        return _BaseVarExtractionInsights.remove_folder(file_path)

    def get_var_analysis_report(self, data, varlist, dep=None, iv_cut=0.01):
        if dep is None:
            dep = self.dep
        self.failed_variables = []
        adapter = as_woe_engine(self.woe_binner) if self.woe_binner is not None else _make_monotone_adapter(self, data, varlist)
        if adapter is None:
            result = self._base.get_var_analysis_report(data, varlist, dep, iv_cut)
            self.failed_variables = getattr(self._base, "failed_variables", [])
            return result
        result = _screening_summary_from_engine(
            data,
            varlist,
            dep,
            adapter,
            iv_cut,
            self.missing_rate_ref,
            failed_variables=self.failed_variables,
        )
        if self.failed_variables:
            failed = [name for name, _ in self.failed_variables]
            warnings.warn(
                f"{len(failed)}/{len(varlist)} variables failed WOE-engine insight computation: "
                f"{failed[:10]}{'...' if len(failed) > 10 else ''} — see smf_logger for details",
                UserWarning,
                stacklevel=2,
            )
        return result

    def plot_woe(self, data, varlist, plot_group=None, plot_dirname="var_analysis_plot", plot_path=None):
        adapter = as_woe_engine(self.woe_binner) if self.woe_binner is not None else _make_monotone_adapter(self, data, varlist)
        if adapter is None:
            return self._base.plot_woe(data, varlist, plot_group, plot_dirname, plot_path)
        if plot_path is None:
            plot_path = self.plot_path
        if plot_path is None:
            return None
        if adapter.get_engine_name() == "monotone" and hasattr(adapter.engine, "plot_woe_graph"):
            import os
            graph_path = os.path.join(plot_path, plot_dirname)
            adapter.engine.plot_woe_graph(graph_path, group_name=plot_group, _df_for_group=data if plot_group else None)
            return None
        return self._base.plot_woe(data, varlist, plot_group, plot_dirname, plot_path)


class CorrelationFilter:
    def __init__(
        self,
        data,
        dep,
        corr_cutpoint=0.8,
        method="pearson",
        tree_binning=False,
        chi2_method=False,
        seed=42,
        chi2_p=0.999,
        init_equi_bins=1000,
        missing_rate_ref=-9999999,
        spec_values=[],
        base_metric="iv",
        woe_engine="master",
        woe_binner=None,
        woe_engine_params=None,
    ):
        self._base = _BaseCorrelationFilter(
            data, dep, corr_cutpoint, method, tree_binning, chi2_method, seed,
            chi2_p, init_equi_bins, missing_rate_ref, spec_values, base_metric,
        )
        self.data = data
        self.dep = dep
        self.corr_cutpoint = corr_cutpoint
        self.method = method
        self.tree_binning = tree_binning
        self.chi2_method = chi2_method
        self.seed = seed
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.missing_rate_ref = missing_rate_ref
        self.spec_values = spec_values
        self.base_metric = base_metric
        self.woe_engine = woe_engine
        self.woe_binner = woe_binner
        self.woe_engine_params = woe_engine_params or {}
        self.correlated_dict = {}
        self.filtered_varlist = []
        self._corr_matrix_cache = None
        self._metric_summary_cache = None

    def _high_corr_pairs(self, varlist):
        if (
            self._corr_matrix_cache is None
            or not set(varlist).issubset(self._corr_matrix_cache.columns)
        ):
            self._corr_matrix_cache = self.data[varlist].corr(method=self.method)
        matrix = self._corr_matrix_cache.reindex(index=varlist, columns=varlist)
        values = matrix.to_numpy(dtype=float)
        row_idx, col_idx = np.triu_indices(len(varlist), k=1)
        pair_values = values[row_idx, col_idx]
        keep = np.isfinite(pair_values) & (np.abs(pair_values) > self.corr_cutpoint)
        names = np.asarray(varlist, dtype=object)
        return pd.DataFrame(
            {
                "VAR1": names[row_idx[keep]],
                "VAR2": names[col_idx[keep]],
                "CORR": pair_values[keep],
            },
            columns=["VAR1", "VAR2", "CORR"],
        )

    def _metric_summary(self, varlist):
        cached_vars = (
            set(self._metric_summary_cache["var"])
            if self._metric_summary_cache is not None and "var" in self._metric_summary_cache
            else set()
        )
        if not set(varlist).issubset(cached_vars):
            insights = VarExtractionInsights(
                self.data,
                self.dep,
                None,
                chi2_method=self.chi2_method,
                chi2_p=self.chi2_p,
                init_equi_bins=self.init_equi_bins,
                tree_binning=self.tree_binning,
                seed=self.seed,
                missing_rate_ref=self.missing_rate_ref,
                spec_values=self.spec_values,
                woe_engine=self.woe_engine,
                woe_binner=self.woe_binner,
                woe_engine_params=self.woe_engine_params,
            )
            self._metric_summary_cache = insights.get_var_analysis_report(
                self.data,
                varlist,
                dep=self.dep,
                iv_cut=0,
            )
        return self._metric_summary_cache

    def __getattr__(self, name):
        return getattr(self._base, name)

    @staticmethod
    def calculate_vif(df):
        return _BaseCorrelationFilter.calculate_vif(df)

    def filter_single_iteration(self, varlist):
        if self.woe_binner is None and self.woe_engine == "master":
            return self._base.filter_single_iteration(varlist)

        name_mapping = {"iv": "iv", "ks": "ks_in_gains"}
        high_corr_var = self._high_corr_pairs(varlist)
        if len(high_corr_var) == 0:
            return varlist

        selected_varlist = []
        removed_varlist = []
        for var in tqdm(high_corr_var["VAR1"].drop_duplicates().tolist()):
            if var in set(removed_varlist + selected_varlist):
                continue
            single_var_corr = high_corr_var.loc[high_corr_var["VAR1"].eq(var)]
            correlated_list = [var] + single_var_corr["VAR2"].drop_duplicates().tolist()
            metric_summary = self._metric_summary(varlist)
            summary = metric_summary[metric_summary["var"].isin(correlated_list)].copy()
            if summary.empty:
                continue
            selected = summary.sort_values([name_mapping[self.base_metric.lower()]], ascending=False)["var"].iloc[0]
            if selected not in selected_varlist:
                selected_varlist.append(selected)
            removed_varlist += [x for x in correlated_list if x != selected and x not in removed_varlist]
            self.correlated_dict[var] = {"corr": single_var_corr, "gains": summary}

        return selected_varlist + [x for x in varlist if x not in (selected_varlist + removed_varlist)]

    def remove_highly_correlated(self, varlist, max_iterations=10):
        if self.woe_binner is None and self.woe_engine == "master":
            result = self._base.remove_highly_correlated(varlist, max_iterations)
            self.correlated_dict = getattr(self._base, "correlated_dict", {})
            self.filtered_varlist = getattr(self._base, "filtered_varlist", [])
            return result

        self._corr_matrix_cache = self.data[varlist].corr(method=self.method)
        self._metric_summary_cache = None
        self._metric_summary(varlist)
        last_keep_list = self.filter_single_iteration(varlist)
        for _ in range(1, max_iterations):
            keep_list = self.filter_single_iteration(last_keep_list)
            removed_vars = [x for x in last_keep_list if x not in keep_list]
            self.filtered_varlist.append(removed_vars)
            if len(removed_vars) == 0:
                break
            last_keep_list = keep_list
        self.filtered_varlist = [x for x in varlist if x not in last_keep_list]
        return last_keep_list


__all__ = ["PSICalculator", "VarExtractionInsights", "CorrelationFilter"]
