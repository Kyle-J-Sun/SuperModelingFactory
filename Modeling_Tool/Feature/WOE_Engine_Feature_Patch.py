"""Compatibility patch for WOE-engine-aware feature screening.

This module extends existing public classes without changing their default
behavior.  When no WOE engine is supplied, all methods delegate to the original
implementation.  When a fitted ``WOE_Master`` or ``MonotoneWOEBinner`` is
provided, PSI/IV/correlation decisions reuse that fitted binning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from Modeling_Tool.WOE.WOE_Adapter import as_woe_engine
from .PSI_Tool import PSICalculator
from .Feature_Insights import CorrelationFilter, VarExtractionInsights, var_corr_filter
from .Distribution_Tool import proc_means_by_grp


_ORIG_PSI_INIT = PSICalculator.__init__
_ORIG_PSI_CALCULATE = PSICalculator.calculate
_ORIG_INSIGHTS_INIT = VarExtractionInsights.__init__
_ORIG_INSIGHTS_REPORT = VarExtractionInsights.get_var_analysis_report
_ORIG_INSIGHTS_PLOT_WOE = VarExtractionInsights.plot_woe
_ORIG_CORR_INIT = CorrelationFilter.__init__
_ORIG_CORR_FILTER_SINGLE = CorrelationFilter.filter_single_iteration


def _psi_from_bins(expected_bins: pd.Series, current_bins: pd.Series, content: float) -> float:
    expected_pct = expected_bins.value_counts(normalize=True, dropna=False)
    current_pct = current_bins.value_counts(normalize=True, dropna=False)
    all_bins = expected_pct.index.union(current_pct.index)
    e = expected_pct.reindex(all_bins, fill_value=0).astype(float) + content
    c = current_pct.reindex(all_bins, fill_value=0).astype(float) + content
    return float(((c - e) * np.log(c / e)).sum())


def _make_adapter_from_insights(self, data: pd.DataFrame, varlist: list[str]):
    if getattr(self, "woe_binner", None) is not None:
        return as_woe_engine(self.woe_binner)

    if getattr(self, "woe_engine", "master") != "monotone":
        return None

    from Modeling_Tool.WOE.WOE_Monotone_Binner import MonotoneWOEBinner

    params = dict(getattr(self, "woe_engine_params", {}) or {})
    fit_params = params.pop("fit_params", {}) if "fit_params" in params else {}
    binner = MonotoneWOEBinner(
        feature_cols=varlist,
        target_col=self.dep,
        special_values=self.spec_values,
        **params,
    )
    binner.fit(
        data,
        chi2_binning=self.chi2_method,
        chi2_p=self.chi2_p,
        chi2_init_size=self.init_equi_bins,
        **fit_params,
    )
    self.woe_binner = binner
    return as_woe_engine(binner)


def _screening_summary_from_engine(
    data: pd.DataFrame,
    varlist: list[str],
    dep: str,
    adapter,
    iv_cut: float,
    missing_rate_ref,
) -> pd.DataFrame:
    rows = []
    target = data[dep]
    total_bad = max(float((target == 1).sum()), 1.0)
    total_good = max(float((target == 0).sum()), 1.0)

    for var in tqdm(varlist):
        if var not in data.columns or data[var].nunique(dropna=False) <= 1:
            continue
        try:
            bins = adapter.assign_bins(data, var)
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
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=[
            "var", "n_all", "n", "ks_in_gains", "lift_in_gains", "iv",
            "n_bump", "missing_rate", "min", "mean", "max", "n_bins",
        ])

    high_iv_summary = pd.DataFrame(rows).query(f"iv >= {iv_cut}").round(4)
    high_iv_varlist = high_iv_summary["var"].tolist()
    if high_iv_varlist:
        means = proc_means_by_grp(data, high_iv_varlist, spec_missing_value=missing_rate_ref)
        summary = high_iv_summary.merge(
            means[["attribute", "N_ALL", "N", "MISSING_RATE", "MIN", "MEAN", "MAX"]],
            left_on="var",
            right_on="attribute",
            how="left",
        )
    else:
        summary = high_iv_summary.copy()
        for col in ["N_ALL", "N", "MISSING_RATE", "MIN", "MEAN", "MAX"]:
            summary[col] = np.nan

    summary.columns = [str(c).lower() for c in summary.columns]
    return summary[[
        "var", "n_all", "n", "ks_in_gains", "lift_in_gains", "iv",
        "n_bump", "missing_rate", "min", "mean", "max", "n_bins",
    ]]


def _psi_init_with_engine(
    self,
    buckets: int = 10,
    equal_freq: bool = True,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    binning_engine=None,
):
    _ORIG_PSI_INIT(self, buckets=buckets, equal_freq=equal_freq, min_bin_prop=min_bin_prop, content=content, precision=precision)
    self.binning_engine = binning_engine
    self._woe_engine_adapter = as_woe_engine(binning_engine) if binning_engine is not None else None


def _psi_calculate_with_engine(
    self,
    expected_df: pd.DataFrame,
    current_data: pd.DataFrame,
    varlist: list[str],
    group_by=None,
    group_name=None,
    return_details=False,
):
    adapter = getattr(self, "_woe_engine_adapter", None)
    if adapter is None:
        return _ORIG_PSI_CALCULATE(self, expected_df, current_data, varlist, group_by, group_name, return_details)

    detail = {}
    rows = []

    if group_by is None:
        groups = [(None, current_data)]
    else:
        groups = list(current_data.groupby(group_by))

    for var in varlist:
        expected_bins = adapter.assign_bins(expected_df, var)
        for grp_value, grp_df in groups:
            current_bins = adapter.assign_bins(grp_df, var)
            psi_value = _psi_from_bins(expected_bins, current_bins, self.content)
            row = {"var": var, "psi": round(psi_value, self.precision)}
            if group_by is not None:
                row[group_name or group_by] = grp_value
            rows.append(row)
            if return_details:
                detail_key = var if grp_value is None else (var, grp_value)
                detail[detail_key] = {
                    "expected_bins": expected_bins.value_counts(normalize=True, dropna=False),
                    "current_bins": current_bins.value_counts(normalize=True, dropna=False),
                }

    result = pd.DataFrame(rows)
    return (result, detail) if return_details else result


def _insights_init_with_engine(
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
    _ORIG_INSIGHTS_INIT(
        self, data, dep, plot_path, nbins, equal_freq, min_bin_prop, precision,
        chi2_method, chi2_p, init_equi_bins, tree_binning, include_missing,
        seed, missing_rate_ref, spec_values,
    )
    self.woe_engine = woe_engine
    self.woe_binner = woe_binner
    self.woe_engine_params = woe_engine_params or {}


def _insights_report_with_engine(self, data, varlist, dep=None, iv_cut=0.01):
    if dep is None:
        dep = self.dep
    adapter = _make_adapter_from_insights(self, data, varlist)
    if adapter is None:
        return _ORIG_INSIGHTS_REPORT(self, data, varlist, dep, iv_cut)
    return _screening_summary_from_engine(data, varlist, dep, adapter, iv_cut, self.missing_rate_ref)


def _insights_plot_woe_with_engine(self, data, varlist, plot_group=None, plot_dirname="var_analysis_plot", plot_path=None):
    adapter = _make_adapter_from_insights(self, data, varlist)
    if adapter is None:
        return _ORIG_INSIGHTS_PLOT_WOE(self, data, varlist, plot_group, plot_dirname, plot_path)

    if plot_path is None:
        plot_path = self.plot_path
    if plot_path is None:
        return None

    if adapter.get_engine_name() == "monotone" and hasattr(adapter.engine, "plot_woe_graph"):
        import os
        graph_path = os.path.join(plot_path, plot_dirname)
        adapter.engine.plot_woe_graph(graph_path, group_name=plot_group, _df_for_group=data if plot_group else None)
        return None

    return _ORIG_INSIGHTS_PLOT_WOE(self, data, varlist, plot_group, plot_dirname, plot_path)


def _corr_init_with_engine(
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
    _ORIG_CORR_INIT(
        self, data, dep, corr_cutpoint, method, tree_binning, chi2_method, seed,
        chi2_p, init_equi_bins, missing_rate_ref, spec_values, base_metric,
    )
    self.woe_engine = woe_engine
    self.woe_binner = woe_binner
    self.woe_engine_params = woe_engine_params or {}


def _corr_filter_single_with_engine(self, varlist):
    if getattr(self, "woe_binner", None) is None and getattr(self, "woe_engine", "master") == "master":
        return _ORIG_CORR_FILTER_SINGLE(self, varlist)

    base_metric = self.base_metric.lower()
    name_mapping = {"iv": "iv", "ks": "ks_in_gains"}
    high_corr_var = var_corr_filter(self.data, varlist, corr_cutpoint=self.corr_cutpoint, method=self.method)
    if len(high_corr_var) == 0:
        return varlist

    base_varlist = high_corr_var["VAR1"].drop_duplicates().tolist()
    selected_varlist = []
    removed_varlist = []

    for var in tqdm(base_varlist):
        if var in set(removed_varlist + selected_varlist):
            continue
        single_var_corr = high_corr_var.query(f"VAR1 == '{var}'")
        correlated_list = [var] + single_var_corr["VAR2"].drop_duplicates().tolist()
        insights = VarExtractionInsights(
            data=self.data,
            dep=self.dep,
            plot_path=None,
            nbins=10,
            equal_freq=True,
            min_bin_prop=0.05,
            precision=5,
            chi2_method=self.chi2_method,
            chi2_p=self.chi2_p,
            init_equi_bins=self.init_equi_bins,
            tree_binning=self.tree_binning,
            include_missing=True,
            seed=self.seed,
            missing_rate_ref=self.missing_rate_ref,
            spec_values=self.spec_values,
            woe_engine=self.woe_engine,
            woe_binner=self.woe_binner,
            woe_engine_params=self.woe_engine_params,
        )
        summary = insights.get_var_analysis_report(self.data, correlated_list, dep=self.dep, iv_cut=0)
        if summary.empty:
            continue
        selected = summary.sort_values([name_mapping[base_metric]], ascending=False)["var"].iloc[0]
        if selected not in selected_varlist:
            selected_varlist.append(selected)
        removed_varlist += [x for x in correlated_list if x != selected and x not in removed_varlist]
        self.correlated_dict[var] = {"corr": single_var_corr, "gains": summary}

    other_varlist = [x for x in varlist if x not in (selected_varlist + removed_varlist)]
    return selected_varlist + other_varlist


PSICalculator.__init__ = _psi_init_with_engine
PSICalculator.calculate = _psi_calculate_with_engine
VarExtractionInsights.__init__ = _insights_init_with_engine
VarExtractionInsights.get_var_analysis_report = _insights_report_with_engine
VarExtractionInsights.plot_woe = _insights_plot_woe_with_engine
CorrelationFilter.__init__ = _corr_init_with_engine
CorrelationFilter.filter_single_iteration = _corr_filter_single_with_engine

__all__ = []
