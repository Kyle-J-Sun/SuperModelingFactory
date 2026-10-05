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
    """Population Stability Index (PSI) calculator that can bin with a fitted WOE engine.

    Without ``binning_engine`` every call is delegated to ``Modeling_Tool.Feature.PSI_Tool.PSICalculator``: each
    variable is binned on the expected sample (equal-frequency or equal-width bins) and the current sample is
    cut with the same edges. With a ``binning_engine`` the PSI is computed on the bins of that engine instead
    (its WOE value is the bin label), so monitoring uses the same mapping as the model.

    Parameters
    ----------
    buckets : int, default 10
        Number of bins of the default binning. It has no effect when ``binning_engine`` is given.
    equal_freq : bool, default True
        Use equal-frequency bins (False gives equal-width bins). It has no effect when ``binning_engine`` is given.
    min_bin_prop : float, default 0.05
        Minimum proportion for each bin; it caps the number of bins at ``max(5, 1 / min_bin_prop)``. It has no
        effect when ``binning_engine`` is given.
    content : float, default 1e-06
        Floor of a bin share, so that the logarithm stays finite (see ``psi_missing_bucket_policy``).
    precision : int, default 5
        Decimals of the bin edges of the default binning. With ``binning_engine`` it is the number of decimals
        the PSI values are rounded to.
    binning_engine : WOE_Master, MonotoneWOEBinner, WOEEngineAdapter or None, default None
        Fitted WOE engine whose bins define the PSI. ``None`` keeps the default binning. An object of any other type
        raises ``TypeError``.
    missing_policy : {"include", "drop", "warn_and_drop"}, default "include"
        How NaN rows are handled by the default binning: ``"include"`` gives them a bin of their own
        (``"__MISSING__"``), so a change of the missing rate moves the PSI; ``"drop"`` ignores them;
        ``"warn_and_drop"`` drops them and emits a ``RuntimeWarning`` with the NaN counts. It has no effect when
        ``binning_engine`` is given (the engine decides how missing values are binned).
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, default "smooth_laplace"
        How a bin that holds rows on one side only is treated. ``"smooth_laplace"`` adds one to every bin count
        when such a bin exists; ``"floor_1e6"`` clips the bin shares at ``content``; ``"exclude"`` drops the
        one-sided bins from the sum. Any other value raises ``ValueError``.
    feature_block_size : int or None, default 64
        Number of variables that the engine transforms per block (used only with ``binning_engine``). ``None``
        processes all variables at once. A value that is not a positive integer raises ``ValueError``.

    Attributes
    ----------
    buckets, equal_freq, min_bin_prop, content, precision, missing_policy, psi_missing_bucket_policy, feature_block_size
        The constructor values.
    binning_engine : WOE_Master, MonotoneWOEBinner, WOEEngineAdapter or None
        The engine as passed to the constructor.

    Examples
    --------
    >>> psi = PSICalculator(buckets=10, binning_engine=woe).calculate(train_df, oot_df, features)
    """

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
        """Calculate the PSI of each variable between ``expected_df`` and ``current_data``.

        Parameters
        ----------
        expected_df : pandas.DataFrame
            Expected (baseline) sample. The default binning builds the bins on it.
        current_data : pandas.DataFrame
            Current sample that is compared with ``expected_df``.
        varlist : list of str
            Variables to compute the PSI for. With a ``binning_engine`` every variable must be one the engine was
            fitted on, otherwise ``KeyError`` is raised.
        group_by : str, list of str or None, default None
            Legacy grouping argument. With the default binning it has no effect unless ``group_name`` is also given
            or ``return_details`` is True; use ``group_name`` to get PSI by group. With a ``binning_engine`` it acts
            as ``group_name`` when ``group_name`` is None.
        group_name : str, list of str or None, default None
            Column(s) of ``current_data`` whose values define groups. Every group is compared with the whole
            ``expected_df`` and the bins are built once, on ``expected_df``. A NaN group value becomes ``"__NULL__"``.
            A list of columns is supported only with a ``binning_engine``.
        return_details : bool, default False
            Whether to return the bin-level details together with the PSI table (see Returns).
        missing_policy : {"include", "drop", "warn_and_drop"} or None, default None
            How NaN rows are handled by the default binning. ``None`` uses the value of the constructor. It has no
            effect when a ``binning_engine`` is used.
        psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"} or None, default None
            How a bin that holds rows on one side only is treated. ``None`` uses the value of the constructor.

        Returns
        -------
        pandas.DataFrame, dict or tuple
            The PSI table has one row per variable (and per group value) with the columns ``var`` and ``psi`` plus
            the group column(s). With ``return_details=True``:

            - default binning: the dict ``{'psi': PSI table, 'details': DataFrame}``, where ``details`` has one row
              per bin with the columns ``bin``, ``expected_count``, ``actual_count``, ``expected_percent``,
              ``actual_percent``, ``psi_component``, ``bucket_status``, ``psi_missing_bucket_policy``, the group
              column when ``group_name`` is given, and ``var``;
            - ``binning_engine``: the tuple ``(PSI table, details)``, where ``details`` maps ``var`` (or
              ``(var, group value)`` when grouped) to ``{'expected_bins': Series, 'current_bins': Series}``, the
              share of rows per bin sorted in descending order.

        Raises
        ------
        ValueError
            If ``psi_missing_bucket_policy`` is not a valid value (or, with the default binning, ``missing_policy``).
        KeyError
            With a ``binning_engine``, if a variable of ``varlist`` was not fitted by the engine.

        Notes
        -----
        With a ``binning_engine`` the PSI values are rounded to ``precision`` decimals. The default binning does not
        round them.
        """
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
    """Variable insight analyzer: IV, KS and lift per variable, and bivariate WOE charts.

    Without an engine the calls are delegated to ``Modeling_Tool.Feature.Feature_Insights.VarExtractionInsights``, which
    bins every variable by its own settings (decision-tree bins by default). With ``woe_binner`` (a fitted ``WOE_Master``
    or ``MonotoneWOEBinner``) or with ``woe_engine="monotone"``, the bins and the IV come from that engine instead.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data. It is stored on the instance, and every method takes its own ``data`` argument.
    dep : str
        Name of the target column (0/1).
    plot_path : str or None
        Root folder of the charts written by ``plot_woe``. Pass ``None`` if you do not plot.
        ``get_var_analysis_report`` writes no files.
    nbins : int, default 10
        Number of bins per variable.
    equal_freq : bool, default True
        Use equal-frequency bins (False gives equal-width bins).
    min_bin_prop : float, default 0.05
        Minimum proportion of rows per bin.
    precision : int, default 5
        Decimals of the bin edges.
    chi2_method : bool, default False
        Merge an initial fine binning with a chi-square test.
    chi2_p : float, default 0.9
        Confidence level of the chi-square merge.
    init_equi_bins : int, default 5000
        Number of fine bins before the chi-square merge.
    tree_binning : bool, default True
        Place the bin edges with a decision tree on the target (supervised bins).
    include_missing : bool, default True
        Missing values form their own bin.
    seed : int, default 3407
        Random seed of the tree binning.
    missing_rate_ref : int or float, default -999999
        Sentinel for missing values. It fills ``NaN`` in the default binning and is treated as missing in the
        ``missing_rate``, ``min``, ``mean`` and ``max`` columns of the report.
    spec_values : list or None, default None
        Special values that get their own bins. ``None`` is stored as an empty list.
    woe_engine : str, default "master"
        ``"monotone"`` makes the object fit its own ``MonotoneWOEBinner`` on first use and keep it in
        ``woe_binner``. Any other value behaves like ``"master"`` (no self-fitted engine).
    woe_binner : WOE_Master, MonotoneWOEBinner or None, default None
        A fitted engine. It is used whenever it is given, whatever ``woe_engine`` says: bins and IV then come from
        it, and ``nbins``, ``equal_freq``, ``min_bin_prop``, ``precision``, ``chi2_method``, ``chi2_p``,
        ``init_equi_bins``, ``tree_binning``, ``include_missing``, ``seed`` and ``spec_values`` are ignored by the
        report. An object of another type raises ``TypeError`` when a method uses it.
    woe_engine_params : dict or None, default None
        Constructor arguments of the self-fitted ``MonotoneWOEBinner`` (``woe_engine="monotone"`` without
        ``woe_binner``). The key ``fit_params`` is passed on to its ``fit``. ``chi2_method``, ``chi2_p``,
        ``init_equi_bins`` and ``spec_values`` also feed that fit. ``None`` is stored as an empty dict.

    Attributes
    ----------
    data, dep, plot_path, nbins, equal_freq, min_bin_prop, precision, chi2_method, chi2_p, init_equi_bins, tree_binning, include_missing, seed, missing_rate_ref, spec_values, woe_engine, woe_engine_params
        The constructor values.
    woe_binner : WOE_Master, MonotoneWOEBinner or None
        The engine of the constructor, or the self-fitted ``MonotoneWOEBinner`` after the first call that needed it.
    failed_variables : list of tuple
        ``(variable, exception type name)`` pairs of the variables that the last ``get_var_analysis_report`` call
        could not analyze.

    Examples
    --------
    >>> insights = VarExtractionInsights(train_df, "bad_flag", None, woe_binner=woe)
    >>> report = insights.get_var_analysis_report(train_df, features, iv_cut=0)
    """

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
        """Delete the specified folder.

        Recursively delete the folder at the given path together with all of its
        contents; do nothing (silently) if the folder does not exist.

        Parameters
        ----------
        file_path : str
            Path of the folder to delete.
        """
        return _BaseVarExtractionInsights.remove_folder(file_path)

    def get_var_analysis_report(self, data, varlist, dep=None, iv_cut=0.01):
        """Compute IV, KS and lift of each variable and return those that reach ``iv_cut``.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the variables and the target column.
        varlist : list of str
            Variables to analyze.
        dep : str or None, default None
            Name of the target column. ``None`` uses the ``dep`` of the constructor. The argument is honored only
            when a WOE engine supplies the bins (``woe_binner``, or ``woe_engine="monotone"``): the default binning
            always uses the ``dep`` of the constructor.
        iv_cut : float, default 0.01
            IV threshold. Variables with ``iv < iv_cut`` are left out of the report; pass 0 to keep them all.

        Returns
        -------
        pandas.DataFrame
            One row per reported variable with the columns ``var``, ``n_all`` (rows), ``n`` (non-missing rows),
            ``ks_in_gains``, ``lift_in_gains`` (largest bin lift), ``iv`` (rounded to 4 decimals), ``n_bump``,
            ``n_bins``, ``missing_rate``, ``min``, ``mean`` and ``max``. ``n_bump`` counts the rank-order breaks
            between the bins of the default binning, and equals the bin count when a WOE engine is used. The
            default binning sorts the rows by ``iv``, descending; with a WOE engine they follow ``varlist``. The
            frame is empty (with these columns) when no variable could be analyzed.

        Warns
        -----
        UserWarning
            If some variables could not be analyzed (they are listed in ``failed_variables``).

        Notes
        -----
        A constant column is skipped silently. A variable that cannot be binned is left out and recorded in
        ``failed_variables`` as ``(name, exception type)``. With a WOE engine, a variable that is missing from
        ``data`` is skipped silently, and a variable that the engine was not fitted on is recorded as failed (a
        self-fitted ``MonotoneWOEBinner`` is fitted once, on the ``varlist`` of the first call). The default binning
        handles numeric variables only.
        """
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
        """Plot the bivariate WOE charts of the variables and save them as PNG files.

        Parameters
        ----------
        data : pandas.DataFrame
            Data that holds the variables and the target column (and ``plot_group`` when it is given).
        varlist : list of str
            Variables to plot. When the object holds a ``MonotoneWOEBinner`` (given as ``woe_binner`` or
            self-fitted), every variable that the binner was fitted on is plotted and ``varlist`` is ignored.
        plot_group : str or None, default None
            Name of a grouping column. A chart per variable split by this column is written as well
            (``<var>_<plot_group>.png``). With a ``MonotoneWOEBinner`` the charts of the groups
            (``<var>_by_<plot_group>.png``) replace the ungrouped ones. The group values must be strings in the
            default plot (numeric values raise ``TypeError``).
        plot_dirname : str, default "var_analysis_plot"
            Subdirectory of ``plot_path`` that receives the charts. The folders are created if they do not exist.
        plot_path : str or None, default None
            Root folder of the charts. ``None`` uses the ``plot_path`` of the constructor.

        Returns
        -------
        None
            Saves the images directly.

        Raises
        ------
        TypeError
            If no plot path is available (``plot_path`` and the constructor's ``plot_path`` are both None) while the
            default binning is used. With a ``MonotoneWOEBinner`` the method returns without plotting instead.

        Notes
        -----
        Only a ``MonotoneWOEBinner`` plots from its own bins. Without an engine, or with a ``WOE_Master`` as
        ``woe_binner``, a new ``WOE_Master`` is fitted with the binning settings of the constructor (missing values
        are first filled with ``missing_rate_ref``) and the charts are drawn from it.
        """
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
    """Correlation filter: drop the weaker variable of each highly correlated pair.

    Pairs whose absolute correlation exceeds ``corr_cutpoint`` are found, and from each group of correlated variables
    the one with the highest IV (or KS) is kept. Without an engine the calls are delegated to
    ``Modeling_Tool.Feature.Feature_Insights.CorrelationFilter``; with ``woe_binner`` (a fitted ``WOE_Master`` or
    ``MonotoneWOEBinner``) or ``woe_engine="monotone"``, the IV and KS come from that engine.

    Parameters
    ----------
    data : pandas.DataFrame
        Input data (raw values). The correlation is computed on the raw values of the numeric variables.
    dep : str
        Name of the target column (0/1).
    corr_cutpoint : float, default 0.8
        Correlation threshold: a pair is correlated when ``abs(corr) > corr_cutpoint``.
    method : str, default "pearson"
        Correlation method, ``"pearson"``, ``"spearman"`` or ``"kendall"``, passed to ``DataFrame.corr``.
    tree_binning : bool, default False
        Use decision-tree bins in the default IV and KS computation. Ignored when an engine supplies the bins.
    chi2_method : bool, default False
        Merge an initial fine binning with a chi-square test in the default computation. With
        ``woe_engine="monotone"`` it also feeds the self-fitted binner.
    seed : int, default 42
        Random seed of the tree binning. Ignored when an engine supplies the bins.
    chi2_p : float, default 0.999
        Confidence level of the chi-square merge.
    init_equi_bins : int, default 1000
        Number of fine bins before the chi-square merge.
    missing_rate_ref : int or float, default -9999999
        Sentinel for missing values in the default IV and KS computation. Ignored when an engine supplies the bins.
    spec_values : list, default []
        Special values (for example ``-1`` or ``999`` for "no record") that get bins or rows of their own in the IV and
        KS calculation, with the default computation and with an engine alike. The correlation itself is still
        computed on the raw values.
    base_metric : {"iv", "ks"}, default "iv"
        Metric that decides which variable of a correlated group survives. Any other value raises ``KeyError`` when
        a pair is filtered.
    woe_engine : str, default "master"
        ``"monotone"`` without ``woe_binner`` fits a ``MonotoneWOEBinner`` on ``data``. Any other value behaves like
        ``"master"`` (the default IV and KS computation).
    woe_binner : WOE_Master, MonotoneWOEBinner or None, default None
        A fitted engine that supplies the IV and KS. It is used whenever it is given, and it also encodes the
        non-numeric variables for the correlation. An object of another type raises ``TypeError`` when it is used.
    woe_engine_params : dict or None, default None
        Constructor arguments of the self-fitted ``MonotoneWOEBinner``; the key ``fit_params`` is passed on to its
        ``fit``. ``None`` is stored as an empty dict.

    Attributes
    ----------
    data, dep, corr_cutpoint, method, tree_binning, chi2_method, seed, chi2_p, init_equi_bins, missing_rate_ref, spec_values, base_metric, woe_engine, woe_binner, woe_engine_params
        The constructor values.
    correlated_dict : dict
        For each anchor variable of a correlated group, ``{"corr": pairs DataFrame, "gains": metric table}``.
    filtered_varlist : list
        The variables removed by ``remove_highly_correlated``.

    Examples
    --------
    >>> corr_filter = CorrelationFilter(train_df, "bad_flag", corr_cutpoint=0.7, woe_binner=woe)
    >>> keep_vars = corr_filter.remove_highly_correlated(features)
    """

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
        self._corr_matrix_excluded = set()
        self._metric_summary_cache = None
        self._correlation_decision_trace = []

    def _corr_matrix_frame(self, varlist):
        # 0.7.0-R1: mixed correlation base. Numeric cols stay raw so a
        # pure-numeric varlist is byte-identical to the legacy .corr() path;
        # non-numeric cols are WOE-encoded via the screening binner
        # (corr_use_woe_bins semantics) and renamed back to raw names. Cols that
        # cannot be encoded go to self._corr_matrix_excluded: they leave the
        # matrix and survive the corr stage as NaN rows/cols once _high_corr_pairs
        # reindexes them back in (same as the weighted raw precedent). At most one
        # UserWarning per matrix build.
        non_numeric = [c for c in varlist if not pd.api.types.is_numeric_dtype(self.data[c])]
        if not non_numeric:
            self._corr_matrix_excluded = set()
            return self.data[varlist]

        encoded = {}
        excluded = set()
        if self.woe_binner is not None:
            suffix = str(getattr(self.woe_binner, "woe_suffix", "_woe"))
            try:
                adapter = as_woe_engine(self.woe_binner, woe_suffix=suffix)
            except TypeError as exc:
                raise TypeError(
                    "corr_use_woe_bins=True routed categorical feature(s) through "
                    "the screening WOE engine at the corr stage, but the supplied "
                    "woe_binner is an Unsupported WOE engine. Expected WOE_Master, "
                    "MonotoneWOEBinner, or WOEEngineAdapter."
                ) from exc
            suffix = adapter.woe_suffix
            try:
                tx = adapter.transform(self.data.copy(), varlist=non_numeric, suffix=suffix)
            except (TypeError, ValueError, KeyError, AttributeError, np.linalg.LinAlgError):
                tx = None
            for name in non_numeric:
                column = f"{name}{suffix}"
                if tx is not None and column in tx.columns and pd.api.types.is_numeric_dtype(tx[column]):
                    encoded[name] = tx[column].to_numpy()
                else:
                    excluded.add(name)
            if excluded:
                excluded_list = [c for c in non_numeric if c in excluded]
                warnings.warn(
                    f"corr_use_woe_bins=True could not WOE-encode {len(excluded)} "
                    f"feature(s) {excluded_list[:5]}; they are excluded from the "
                    f"correlation matrix and kept through the corr stage. Refit the "
                    f"screening WOE engine to cover them.",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            excluded = set(non_numeric)
            warnings.warn(
                f"raw-value correlation skips {len(non_numeric)} non-numeric "
                f"feature(s) {non_numeric[:5]}; they are kept through the corr "
                f"stage. Set corr_use_woe_bins=True to correlate categorical "
                f"features via their WOE encoding.",
                UserWarning,
                stacklevel=2,
            )

        self._corr_matrix_excluded = excluded
        selected = [c for c in varlist if c not in excluded]
        frame = self.data[selected].copy()
        for name, values in encoded.items():
            frame[name] = values
        return frame

    def _high_corr_pairs(self, varlist):
        if (
            self._corr_matrix_cache is None
            or not (set(varlist) <= set(self._corr_matrix_cache.columns) | self._corr_matrix_excluded)
        ):
            self._corr_matrix_cache = self._corr_matrix_frame(varlist).corr(method=self.method)
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
        """Compute the variance inflation factor (VIF) of each column.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame that holds the independent variables (numeric columns without missing values).

        Returns
        -------
        pandas.DataFrame
            One row per column of ``df`` with the columns ``index`` (the variable name) and ``VIF``. A VIF above 10
            usually signals serious collinearity.

        Notes
        -----
        It needs the optional ``statsmodels`` package. There is no NaN handling and no weights.
        """
        return _BaseCorrelationFilter.calculate_vif(df)

    def _sync_base_state(self):
        self.correlated_dict = getattr(self._base, "correlated_dict", {})
        self.filtered_varlist = getattr(self._base, "filtered_varlist", [])
        self._corr_matrix_cache = getattr(self._base, "_corr_matrix_cache", None)
        self._corr_matrix_excluded = getattr(self._base, "_corr_matrix_excluded", set())
        self._metric_summary_cache = getattr(self._base, "_metric_summary_cache", None)
        self._correlation_decision_trace = getattr(
            self._base, "_correlation_decision_trace", [],
        )

    def filter_single_iteration(self, varlist):
        """Run one pass of the correlation filter over ``varlist``.

        From each group of highly correlated variables the one with the highest ``base_metric`` is kept and the
        others are removed.

        Parameters
        ----------
        varlist : list of str
            Variables to screen. They must be columns of ``data``.

        Returns
        -------
        list of str
            The retained variables: the winner of each correlated group first (in the order of discovery), followed
            by the variables that are not part of any highly correlated pair in their original order. ``varlist``
            itself is returned when no pair exceeds ``corr_cutpoint``.

        Notes
        -----
        The call updates ``correlated_dict`` and the internal decision trace. Non-numeric variables are encoded with
        the WOE engine for the correlation when one is available; otherwise they are skipped with a warning and
        kept.
        """
        if self.woe_binner is None and self.woe_engine == "master":
            result = self._base.filter_single_iteration(varlist)
            self._sync_base_state()
            return result

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
            newly_removed = [
                x for x in correlated_list
                if x != selected and x not in removed_varlist
            ]
            metric_name = name_mapping[self.base_metric.lower()]
            metric_map = summary.set_index("var")[metric_name].to_dict()
            positions = {name: idx for idx, name in enumerate(varlist)}
            for dropped_var in newly_removed:
                decision_pair = [selected, dropped_var]
                decision_pair.sort(key=positions.__getitem__)
                var_a, var_b = decision_pair
                corr_value = self._corr_matrix_cache.loc[var_a, var_b]
                self._correlation_decision_trace.append({
                    "var_a": var_a,
                    "var_b": var_b,
                    "corr": float(corr_value),
                    "iv_a": float(metric_map.get(var_a, 0.0)),
                    "iv_b": float(metric_map.get(var_b, 0.0)),
                    "kept": selected,
                    "dropped": dropped_var,
                })
            removed_varlist += newly_removed
            self.correlated_dict[var] = {"corr": single_var_corr, "gains": summary}

        return selected_varlist + [x for x in varlist if x not in (selected_varlist + removed_varlist)]

    def remove_highly_correlated(self, varlist, max_iterations=10):
        """Iteratively remove highly correlated variables.

        Run the correlation filter repeatedly until no variable is removed or the maximum number of iterations is
        reached.

        Parameters
        ----------
        varlist : list of str
            Variables to screen. They must be columns of ``data``.
        max_iterations : int, default 10
            Maximum number of filtering passes.

        Returns
        -------
        list of str
            The variables finally retained, with the winner of each correlated group first.

        Notes
        -----
        ``filtered_varlist`` is set to the removed variables, and ``correlated_dict`` holds the correlated pairs and
        the metric table behind each decision.
        """
        if self.woe_binner is None and self.woe_engine == "master":
            result = self._base.remove_highly_correlated(varlist, max_iterations)
            self._sync_base_state()
            return result

        self._correlation_decision_trace = []
        self._corr_matrix_cache = self._corr_matrix_frame(varlist).corr(method=self.method)
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
