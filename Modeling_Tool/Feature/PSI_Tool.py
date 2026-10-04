"""
PSI (Population Stability Index) Calculator Module

This module provides functions and classes for calculating Population Stability Index (PSI)
to measure the distribution drift between expected and actual datasets.

Author: Matrix Agent
"""

import warnings

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Callable, Any
from tqdm import tqdm
from Modeling_Tool.Core.Binning_Tool import quick_binning

# Missing bucket label used across PSI helpers when ``missing_policy="include"``
# (the 0.4.2 default). Kept identical to the sentinel already used by
# ``Feature.Weighted_Screen`` and ``WOE_Adapter`` so downstream aggregators can
# recognise it without special-casing.
_MISSING_BIN = "__MISSING__"
_PSI_BUCKET_POLICIES = {"floor_1e6", "smooth_laplace", "exclude"}


def _validate_psi_bucket_policy(policy: str, caller: str) -> None:
    if policy not in _PSI_BUCKET_POLICIES:
        raise ValueError(
            f"{caller}: psi_missing_bucket_policy must be one of "
            f"{sorted(_PSI_BUCKET_POLICIES)}; got {policy!r}."
        )


def _psi_distributions_from_counts(
    expected_count: pd.Series,
    actual_count: pd.Series,
    expected_total: float,
    actual_total: float,
    *,
    content: float,
    policy: str,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    _validate_psi_bucket_policy(policy, "_psi_distributions_from_counts")
    all_bins = expected_count.index.union(actual_count.index)
    expected_aligned = expected_count.reindex(all_bins, fill_value=0).astype(float)
    actual_aligned = actual_count.reindex(all_bins, fill_value=0).astype(float)
    one_sided = (expected_aligned == 0) | (actual_aligned == 0)

    expected_total = float(expected_total) if expected_total > 0 else 1.0
    actual_total = float(actual_total) if actual_total > 0 else 1.0

    if policy == "exclude":
        keep = ~one_sided
        expected_aligned = expected_aligned[keep]
        actual_aligned = actual_aligned[keep]
        if expected_aligned.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty, empty
        expected_pct = expected_aligned / expected_total
        actual_pct = actual_aligned / actual_total
    elif policy == "smooth_laplace" and bool(one_sided.any()):
        n_buckets = max(len(all_bins), 1)
        expected_pct = (expected_aligned + 1.0) / (expected_total + n_buckets)
        actual_pct = (actual_aligned + 1.0) / (actual_total + n_buckets)
    else:
        expected_pct = (expected_aligned / expected_total).clip(lower=content)
        actual_pct = (actual_aligned / actual_total).clip(lower=content)

    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)

    floor_expected = (expected_aligned / expected_total).clip(lower=content)
    floor_actual = (actual_aligned / actual_total).clip(lower=content)
    floor_values = (floor_actual - floor_expected) * np.log(floor_actual / floor_expected)
    return expected_pct, actual_pct, psi_values, floor_values

# ============================================================================
# Classes
# ============================================================================

class PSICalculator:
    """
    A class for calculating Population Stability Index (PSI) with configurable parameters.
    
    This class encapsulates common PSI calculation parameters and provides methods
    for various PSI calculations including single variable, grouped, and multi-variable
    comparisons between datasets.
    
    Parameters
    ----------
    buckets : int, optional
        Number of bins for binning. Default is 10.
    equal_freq : bool, optional
        Whether to use equal frequency binning. Default is True.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision for results. Default is 5.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default is "include", which routes NaN rows through a dedicated "__MISSING__" bin
        so missing-rate drift contributes to the PSI. See ``__init__`` for the semantics of each mode.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace". See ``__init__``.
    feature_block_size : int or None, optional
        Number of variables processed per block. Default is 64. This class only validates and stores it (a positive
        integer or None); it has no effect on the calculation here.

    Examples
    --------
    >>> calculator = PSICalculator(buckets=10, equal_freq=True)
    >>> psi = calculator.calculate(expected_df, actual_df, 'score')
    """
    
    def __init__(
        self,
        buckets: int = 10,
        equal_freq: bool = True,
        min_bin_prop: float = 0.05,
        content: float = 1e-6,
        precision: int = 5,
        missing_policy: str = "include",
        psi_missing_bucket_policy: str = "smooth_laplace",
        feature_block_size: int | None = 64,
    ):
        """
        Initialize PSICalculator with configuration parameters.
        
        Parameters
        ----------
        buckets : int, optional
            Number of bins for binning. Default is 10.
        equal_freq : bool, optional
            Use equal frequency binning if True. Default is True.
        min_bin_prop : float, optional
            Minimum proportion for each bin. Default is 0.05.
        content : float, optional
            Small value to prevent division by zero. Default is 1e-6.
        precision : int, optional
            Decimal precision for rounding. Default is 5.
        missing_policy : {"drop", "include", "warn_and_drop"}, optional
            How NaN rows are handled. Default in 0.5.0 is "include", which routes
            NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
            contributes to the PSI. This is the recommended production behaviour
            and became the default in 0.5.0 (previously "drop" in 0.4.2 for
            numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
            numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
            naming the NaN counts.
        psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
            How a bin that holds rows on one side only (for example a category that disappears) is treated.
            Default is "smooth_laplace": when such a bin exists, one is added to every bin count before the shares
            are computed. "floor_1e6" clips the bin shares at ``content`` (``1e-6`` by default), so the empty side
            counts as ``content``. "exclude" drops the one-sided bins from the sum. Any other value raises
            ``ValueError``.
        feature_block_size : int or None, optional
            Number of variables processed per block. Default is 64. This class only validates and stores it (a
            positive integer or None, otherwise ``ValueError``); it has no effect on the calculation here.

        Raises
        ------
        ValueError
            If ``missing_policy`` or ``psi_missing_bucket_policy`` is not one of the listed values, or if
            ``feature_block_size`` is not None and not a positive integer.
        """
        if missing_policy not in {"include", "drop", "warn_and_drop"}:
            raise ValueError(
                f"PSICalculator.__init__: missing_policy must be one of "
                f"'include', 'drop', 'warn_and_drop'; got {missing_policy!r}."
            )
        _validate_psi_bucket_policy(psi_missing_bucket_policy, "PSICalculator.__init__")
        if feature_block_size is not None and int(feature_block_size) <= 0:
            raise ValueError("feature_block_size must be a positive integer or None")
        self.buckets = buckets
        self.equal_freq = equal_freq
        self.min_bin_prop = min_bin_prop
        self.content = content
        self.precision = precision
        self.missing_policy = missing_policy
        self.psi_missing_bucket_policy = psi_missing_bucket_policy
        self.feature_block_size = feature_block_size
    
#     def _calculate_single_psi(
#         self,
#         expected_series: pd.Series,
#         actual_series: pd.Series,
#         return_details: bool = False
#     ) -> Union[float, Tuple[float, pd.DataFrame]]:
#         """
#         Calculate PSI for a single variable.
        
#         This method performs binning on both expected and actual series,
#         then calculates the PSI value based on the distribution difference.
        
#         Parameters
#         ----------
#         expected_series : pandas.Series
#             Expected/baseline data series.
#         actual_series : pandas.Series
#             Actual/comparison data series.
#         return_details : bool, optional
#             Whether to return detailed bin information. Default is False.
            
#         Returns
#         -------
#         float or tuple
#             If return_details is False: Returns total PSI value.
#             If return_details is True: Returns tuple of (PSI value, details DataFrame).
#         """
#         # Drop NA
#         expected_clean = expected_series.dropna()
#         actual_clean = actual_series.dropna()
        
#         # Bin the expected data to get breakpoints
#         expected_bins, breakpoints = quick_binning(
#             pd.DataFrame(expected_clean), 
#             expected_clean.name, 
#             labels=None, 
#             nbins=self.buckets, 
#             precision=self.precision, 
#             equal_freq=self.equal_freq, 
#             right=True, 
#             include_lowest=False,
#             min_bin_prop=self.min_bin_prop, 
#             tree_binning=False, 
#             target=None, 
#             random_state=42
#         )
        
#         # Bin the actual data using the same breakpoints
#         actual_bins, _ = quick_binning(
#             pd.DataFrame(actual_clean), 
#             actual_clean.name, 
#             labels=None, 
#             nbins=list(breakpoints), 
#             precision=self.precision, 
#             equal_freq=self.equal_freq, 
#             right=True, 
#             include_lowest=False,
#             min_bin_prop=self.min_bin_prop, 
#             tree_binning=False, 
#             target=None, 
#             random_state=42
#         )
        
#         # Get bin proportions
#         expected_percents = expected_bins.value_counts(normalize=True, sort=False)
#         actual_percents = actual_bins.value_counts(normalize=True, sort=False)
        
#         # Ensure both series have the same bin indices
#         all_bins = expected_percents.index.union(actual_percents.index)
#         expected_percents = expected_percents.reindex(all_bins, fill_value=self.content)
#         actual_percents = actual_percents.reindex(all_bins, fill_value=self.content)
        
#         # Clip to avoid division by zero
#         expected_percents = expected_percents.clip(lower=self.content)
#         actual_percents = actual_percents.clip(lower=self.content)
        
#         # Calculate PSI
#         psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
#         psi_total = psi_values.sum()
        
#         if return_details:
#             details = pd.DataFrame({
#                 'expected_percent': expected_percents,
#                 'actual_percent': actual_percents,
#                 'psi_component': psi_values
#             })
#             return psi_total, details
#         else:
#             return psi_total
    
#     def calculate_psi(
#         self,
#         expected: Union[pd.DataFrame, pd.Series],
#         actual: Union[pd.DataFrame, pd.Series],
#         target_col: str,
#         group_by: Optional[Union[str, List[str]]] = None,
#         return_details: bool = False
#     ) -> Union[float, pd.DataFrame, Tuple[Dict, Dict]]:
#         """
#         Calculate PSI for a variable, optionally by groups.
        
#         Parameters
#         ----------
#         expected : pandas.DataFrame or pandas.Series
#             Expected/baseline data.
#         actual : pandas.DataFrame or pandas.Series
#             Actual/comparison data.
#         target_col : str
#             Column name to calculate PSI for.
#         group_by : str or list, optional
#             Column(s) to group by. Default is None (no grouping).
#         return_details : bool, optional
#             Whether to return detailed bin information. Default is False.
            
#         Returns
#         -------
#         float, pandas.DataFrame, or tuple
#             PSI value(s) and optionally details.
#         """
#         if group_by is not None:
#             if isinstance(group_by, str):
#                 group_by = [group_by]
            
#             expected_subset = expected[[target_col] + group_by].copy()
#             actual_subset = actual[[target_col] + group_by].copy()
            
#             expected_subset = expected_subset.copy()
#             actual_subset = actual_subset.copy()
            
#             expected_subset['_dataset'] = 'expected'
#             actual_subset['_dataset'] = 'actual'
            
#             combined = pd.concat([expected_subset, actual_subset], ignore_index=True)
            
#             results = {}
#             details_dict = {}
            
#             for group, group_data in combined.groupby(group_by):
#                 expected_group = group_data[group_data['_dataset'] == 'expected'].drop('_dataset', axis=1)
#                 actual_group = group_data[group_data['_dataset'] == 'actual'].drop('_dataset', axis=1)
                
#                 if expected_group.shape[0] == 0 or actual_group.shape[0] == 0:
#                     results[group] = 999999
#                     continue
                
#                 if return_details:
#                     psi_value, detail = self._calculate_single_psi(
#                         expected_group[target_col], 
#                         actual_group[target_col],
#                         return_details=True
#                     )
#                     results[group] = psi_value
#                     details_dict[group] = detail
#                 else:
#                     results[group] = self._calculate_single_psi(
#                         expected_group[target_col], 
#                         actual_group[target_col]
#                     )
            
#             if return_details:
#                 return results, details_dict
#             else:
#                 return pd.DataFrame(results, index=['psi']).T
        
#         else:
#             if return_details:
#                 return self._calculate_single_psi(
#                     expected[target_col], 
#                     actual[target_col],
#                     return_details=True
#                 )
#             else:
#                 return self._calculate_single_psi(
#                     expected[target_col], 
#                     actual[target_col]
#                 )
    
#     def calculate_within_psi(
#         self,
#         data: pd.DataFrame,
#         grp_name: str,
#         target_col: str,
#         benchmark: Optional[Any] = None,
#         return_details: bool = False,
#         benchmark_display_name: Optional[str] = None
#     ) -> Union[pd.DataFrame, Dict]:
#         """
#         Calculate PSI values within a single dataset, comparing groups to a benchmark.
        
#         Parameters
#         ----------
#         data : pandas.DataFrame
#             Input dataset containing all groups.
#         grp_name : str
#             Column name for grouping.
#         target_col : str
#             Column name to calculate PSI for.
#         benchmark : str or callable, optional
#             Benchmark group value or filter function. If None, uses first group.
#         return_details : bool, optional
#             Whether to return detailed bin information. Default is False.
#         benchmark_display_name : str, optional
#             Custom name for benchmark in results.
            
#         Returns
#         -------
#         pandas.DataFrame or dict
#             PSI results by group, or dict with 'psi' and 'details' keys.
#         """
#         if callable(benchmark):
#             benchmark_data = data[benchmark(data)]
#         else:
#             benchmark_data = data[data[grp_name] == benchmark] if benchmark is not None else data
        
#         obs_values = [x for x in data[grp_name].unique().tolist() if x != benchmark]
        
#         res_dict = {benchmark_display_name: 0} if benchmark_display_name is not None else {}
#         detail_dict = {}
        
#         for obs_value in obs_values:
#             obs_data = data[data[grp_name] == obs_value]
            
#             if return_details:
#                 psi, details = self.calculate_psi(
#                     benchmark_data, 
#                     obs_data, 
#                     target_col=target_col, 
#                     return_details=True
#                 )
#                 res_dict[obs_value] = psi
#                 detail_dict[obs_value] = details
#             else:
#                 psi = self.calculate_psi(
#                     benchmark_data, 
#                     obs_data, 
#                     target_col=target_col
#                 )
#                 res_dict[obs_value] = psi
        
#         fnl_res = pd.DataFrame(res_dict, index=['psi']).T\
#             .reset_index(drop=False)\
#             .rename(columns={"index": grp_name})
        
#         if return_details:
#             return {"psi": fnl_res, "details": detail_dict}
#         else:
#             return fnl_res
    
#     def calculate_psi_within_dataset(
#         self,
#         data: pd.DataFrame,
#         grp_name: str,
#         varlist: List[str],
#         benchmark: Optional[Any] = None
#     ) -> pd.DataFrame:
#         """
#         Calculate PSI for multiple variables within a dataset.
        
#         Parameters
#         ----------
#         data : pandas.DataFrame
#             Input dataset.
#         grp_name : str
#             Column name for grouping.
#         varlist : list
#             List of variable names to calculate PSI for.
#         benchmark : str or callable, optional
#             Benchmark group value or filter function.
            
#         Returns
#         -------
#         pandas.DataFrame
#             Combined PSI results for all variables.
#         """
#         fnl_psi_res = []
#         for var in tqdm(varlist):
#             single_psi = self.calculate_within_psi(
#                 data=data, 
#                 grp_name=grp_name, 
#                 benchmark=benchmark, 
#                 target_col=var
#             ).sort_values([grp_name]).reset_index(drop=True)
            
#             single_psi['var'] = var
#             fnl_psi_res.append(single_psi)
        
#         return pd.concat(fnl_psi_res)
    
#     def calculate_multivar_psi_two_sets(
#         self,
#         expected_df: pd.DataFrame,
#         actual_df: pd.DataFrame,
#         varlist: List[str],
#         group_by: Optional[Union[str, List[str]]] = None
#     ) -> pd.DataFrame:
#         """
#         Calculate PSI for multiple variables by comparing two datasets.
        
#         Parameters
#         ----------
#         expected_df : pandas.DataFrame
#             Expected/baseline dataset.
#         actual_df : pandas.DataFrame
#             Actual/comparison dataset.
#         varlist : list
#             List of variable names to calculate PSI for.
#         group_by : str or list, optional
#             Column(s) to group by.
            
#         Returns
#         -------
#         pandas.DataFrame
#             PSI results for all variables.
#         """
#         multi_psi_res = []
#         for var in tqdm(varlist):
#             single_psi = self.calculate_psi(
#                 expected=expected_df, 
#                 actual=actual_df, 
#                 target_col=var, 
#                 group_by=group_by
#             )
#             if group_by is None:
#                 single_psi = pd.DataFrame([single_psi], columns=['psi'])
#             single_psi['var'] = var
#             multi_psi_res.append(single_psi)
        
#         return pd.concat(multi_psi_res)
    
    def calculate(
        self,
        expected_df: pd.DataFrame,
        current_data: pd.DataFrame,
        varlist: List[str],
        group_by: Optional[str] = None,
        group_name: Optional[str] = None,
        return_details = False,
        missing_policy: Optional[str] = None,
        psi_missing_bucket_policy: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Calculate grouped PSI comparing two datasets, using expected as benchmark.
        
        Parameters
        ----------
        expected_df : pandas.DataFrame
            Expected/baseline dataset.
        current_data : pandas.DataFrame
            Actual/comparison dataset.
        varlist : list
            List of variable names.
        group_by : str, optional
            Column to group by in both datasets. Default is None. Legacy argument: without ``group_name`` it has no
            effect on the default result (use ``group_name`` to get PSI by group); together with ``group_name`` the
            PSI is computed per ``group_by`` value inside each ``group_name`` group.
        group_name : str, optional
            Specific group column name for multi-group calculation. Default is None. It must be a column of
            ``current_data``: every value (NaN becomes "__NULL__") defines a group that is compared with the whole
            ``expected_df``, and the bins are built once on ``expected_df``.
        return_details : bool, optional
            Whether to return detailed bin information. Default is False. If True, a dict ``{'psi': psi_df,
            'details': details_df}`` is returned, where ``details_df`` has one row per bin (and variable, and group)
            with the columns ``bin``, ``expected_count``, ``actual_count``, ``expected_percent``,
            ``actual_percent``, ``psi_component``, ``bucket_status``, ``psi_missing_bucket_policy`` (and the group
            column) and ``var``.
        missing_policy : {"drop", "include", "warn_and_drop"}, optional
            How NaN rows are handled. If None (default), falls back to the value
            configured on ``self.missing_policy``. Pass an explicit value to
            override the class-level default for this call only. See
            ``PSICalculator.__init__`` for the semantics of each mode.
        psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
            How a bin that holds rows on one side only is treated. If None (default), falls back to the value
            configured on ``self.psi_missing_bucket_policy``. Pass an explicit value to override it for this call
            only. See ``PSICalculator.__init__`` for the semantics of each policy.

        Returns
        -------
        pandas.DataFrame or dict
            Grouped PSI results: a DataFrame with the columns ``var`` and ``psi`` (plus the ``group_name`` column
            when it is given). If ``return_details`` is True, the dict ``{'psi': DataFrame, 'details': DataFrame}``.
        """
        effective_policy = missing_policy if missing_policy is not None else self.missing_policy
        effective_bucket_policy = (
            psi_missing_bucket_policy
            if psi_missing_bucket_policy is not None
            else self.psi_missing_bucket_policy
        )
        return calculate_multigroup_psi_two_sets(
            expected_df = expected_df,
            actual_df = current_data,
            varlist = varlist,
            group_by = group_by,
            buckets = self.buckets,
            equal_freq = self.equal_freq,
            min_bin_prop = self.min_bin_prop,
            content = self.content,
            precision = self.precision,
            group_name = group_name,
            return_details = return_details,
            missing_policy = effective_policy,
            psi_missing_bucket_policy = effective_bucket_policy,
        )


# ============================================================================
# Standalone Functions (Preserved from original code with improvements)
# ============================================================================

def _calculate_single_psi(
    expected_series: pd.Series,
    actual_series: pd.Series,
    buckets: int = 10,
    equal_freq: bool = True,
    return_details: bool = False,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> Union[float, Tuple[float, pd.DataFrame]]:
    """
    Calculate Population Stability Index (PSI) for a single variable.

    This function bins both expected and actual series using the same breakpoints,
    then calculates PSI based on the distribution difference between them.

    Missing handling (0.4.2, N29)
    ------------------------------
    Before 0.4.2, both series were unconditionally passed through ``.dropna()``
    before binning. Any drift in the *missing rate itself* — the single most
    common early signal of a data-pipeline break — was silently invisible in
    the returned PSI. 0.4.2 adds the ``missing_policy`` parameter so callers
    can opt in to the corrected behaviour. The default remains ``"drop"`` for
    strict backward compatibility with the pre-0.4.2 numeric output; the next
    minor release will flip the default to ``"include"``. Pass
    ``missing_policy="include"`` to route NaN rows through a dedicated
    ``"__MISSING__"`` bin on both sides so missing-rate drift contributes to
    the PSI. Pass ``missing_policy="warn_and_drop"`` to keep the old numbers
    but at least surface a ``RuntimeWarning`` naming the two NaN counts.

    Parameters
    ----------
    expected_series : pandas.Series
        Expected/baseline data series.
    actual_series : pandas.Series
        Actual/comparison data series.
    buckets : int, optional
        Number of bins. Default is 10.
    equal_freq : bool, optional
        Use equal frequency binning. Default is True.
    return_details : bool, optional
        Return detailed bin information. Default is False.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. ``"drop"`` (default in 0.4.2) reproduces the
        pre-0.4.2 behaviour and silently excludes NaN rows from both sides
        before binning. ``"include"`` treats NaN as its own ``"__MISSING__"``
        bin so missing-rate drift shows up in the PSI — recommended for any
        production drift monitor, and will become the default in the next
        minor release. ``"warn_and_drop"`` behaves like ``"drop"`` but emits
        a ``RuntimeWarning`` naming the two NaN counts so the caller sees what
        was dropped.

    Returns
    -------
    float or tuple
        If return_details is False: Returns total PSI value.
        If return_details is True: Returns tuple of (PSI value, details DataFrame).

    Notes
    -----
    PSI Formula: Σ (Actual% - Expected%) * ln(Actual% / Expected%)
    A PSI < 0.1 indicates stable population, 0.1-0.25 suggests some change,
    and > 0.25 indicates significant drift.
    """
    if missing_policy not in {"include", "drop", "warn_and_drop"}:
        raise ValueError(
            f"_calculate_single_psi: missing_policy must be one of "
            f"'include', 'drop', 'warn_and_drop'; got {missing_policy!r}."
        )
    _validate_psi_bucket_policy(psi_missing_bucket_policy, "_calculate_single_psi")

    # Record NaN counts before splitting so we can (a) re-attach the missing
    # bin under "include" and (b) warn under "warn_and_drop".
    n_expected = int(len(expected_series))
    n_actual = int(len(actual_series))
    n_expected_na = int(expected_series.isna().sum())
    n_actual_na = int(actual_series.isna().sum())

    if missing_policy == "warn_and_drop" and (n_expected_na or n_actual_na):
        warnings.warn(
            f"_calculate_single_psi: dropping NaN rows before binning. "
            f"expected: {n_expected_na}/{n_expected} NaN, "
            f"actual: {n_actual_na}/{n_actual} NaN. "
            f"Pass missing_policy='include' to treat NaN as its own bin so "
            f"missing-rate drift is captured.",
            RuntimeWarning,
            stacklevel=2,
        )

    expected_clean = expected_series.dropna()
    actual_clean = actual_series.dropna()

    # Bin the expected data to get breakpoints
    expected_bins, breakpoints = quick_binning(
        pd.DataFrame(expected_clean),
        expected_clean.name,
        labels=None,
        nbins=buckets,
        precision=precision,
        equal_freq=equal_freq,
        right=True,
        include_lowest=False,
        min_bin_prop=min_bin_prop,
        tree_binning=False,
        target=None,
        random_state=42
    )

    # Bin the actual data using the same breakpoints
    actual_bins, _ = quick_binning(
        pd.DataFrame(actual_clean),
        actual_clean.name,
        labels=None,
        nbins=list(breakpoints),
        precision=precision,
        equal_freq=equal_freq,
        right=True,
        include_lowest=False,
        min_bin_prop=min_bin_prop,
        tree_binning=False,
        target=None,
        random_state=42
    )

    # Get bin counts (finite-value rows only, from quick_binning)
    expected_count = expected_bins.value_counts(normalize=False, sort=False)
    actual_count = actual_bins.value_counts(normalize=False, sort=False)

    if missing_policy == "include":
        # Re-attach the __MISSING__ bin so that missing-rate drift contributes
        # to the PSI on both sides. The denominators use the *original* row
        # counts (before dropna) so the fractions are directly comparable.
        if n_expected_na > 0:
            expected_count = pd.concat(
                [expected_count, pd.Series({_MISSING_BIN: n_expected_na})]
            )
        if n_actual_na > 0:
            actual_count = pd.concat(
                [actual_count, pd.Series({_MISSING_BIN: n_actual_na})]
            )
        expected_denom = float(n_expected) if n_expected > 0 else 1.0
        actual_denom = float(n_actual) if n_actual > 0 else 1.0
    else:
        # "drop" / "warn_and_drop": legacy behaviour, denominators are the
        # NaN-dropped row counts, so missing-rate drift is invisible.
        expected_denom = float(len(expected_bins)) if len(expected_bins) > 0 else 1.0
        actual_denom = float(len(actual_bins)) if len(actual_bins) > 0 else 1.0

    all_bins = expected_count.index.union(actual_count.index)
    expected_count = expected_count.reindex(all_bins, fill_value=0).astype(float)
    actual_count = actual_count.reindex(all_bins, fill_value=0).astype(float)

    expected_percents, actual_percents, psi_values, psi_floor_values = _psi_distributions_from_counts(
        expected_count,
        actual_count,
        expected_denom,
        actual_denom,
        content=content,
        policy=psi_missing_bucket_policy,
    )
    psi_total = psi_values.sum()
    
    if return_details:
        details = pd.DataFrame({
            'expected_count': expected_count,
            'actual_count': actual_count,
            'expected_percent': expected_percents,
            'actual_percent': actual_percents,
            'psi_component': psi_values,
            'psi_component_floor_1e6': psi_floor_values,
        })
        details["bucket_status"] = "common"
        details.loc[(details["expected_count"] == 0) & (details["actual_count"] > 0), "bucket_status"] = "actual_only"
        details.loc[(details["expected_count"] > 0) & (details["actual_count"] == 0), "bucket_status"] = "expected_only"
        details["psi_missing_bucket_policy"] = psi_missing_bucket_policy
        return psi_total, details
    else:
        return psi_total


def calculate_psi(
    expected: Union[pd.DataFrame, pd.Series],
    actual: Union[pd.DataFrame, pd.Series],
    target_col: str,
    buckets: int = 10,
    equal_freq: bool = True,
    group_by: Optional[Union[str, List[str]]] = None,
    return_details: bool = False,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> Union[float, pd.DataFrame, Tuple[Dict, Dict]]:
    """
    Calculate Population Stability Index (PSI) for a variable, optionally by groups.
    
    This function computes PSI to measure the distribution shift between expected
    (baseline) and actual (comparison) datasets for a specified variable.
    
    Parameters
    ----------
    expected : pandas.DataFrame or pandas.Series
        Expected/baseline data.
    actual : pandas.DataFrame or pandas.Series
        Actual/comparison data.
    target_col : str
        Column name to calculate PSI for.
    buckets : int, optional
        Number of bins. Default is 10.
    equal_freq : bool, optional
        Use equal frequency binning. Default is True.
    group_by : str or list, optional
        Column(s) to group by for stratified PSI calculation. Default is None.
    return_details : bool, optional
        Return detailed bin information. Default is False.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
        naming the NaN counts.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace": when such a bin
        exists, one is added to every bin count before the shares are computed. "floor_1e6" clips the bin shares at
        ``content``, so the empty side counts as ``content``. "exclude" drops the one-sided bins from the sum. Any
        other value raises ``ValueError``.

    Returns
    -------
    float, pandas.DataFrame, or tuple
        - If group_by is None and return_details is False: Single PSI float value.
        - If group_by is None and return_details is True: Tuple of (psi_value, details DataFrame), with one row per
          bin and the columns ``expected_count``, ``actual_count``, ``expected_percent``, ``actual_percent``,
          ``psi_component``, ``psi_component_floor_1e6``, ``bucket_status`` and ``psi_missing_bucket_policy``.
        - If group_by is set and return_details is False: DataFrame with PSI values per group.
        - If group_by is set and return_details is True: Tuple of (results_dict, details_dict).

    Notes
    -----
    With ``group_by``, a group that has rows in only one of the two datasets gets the placeholder PSI 999999 (and no
    details).

    Examples
    --------
    >>> # Simple PSI calculation
    >>> psi = calculate_psi(expected_df, actual_df, 'score')
    
    >>> # PSI by groups
    >>> psi_by_region = calculate_psi(expected_df, actual_df, 'score', group_by='region')
    """
    if group_by is not None:
        if isinstance(group_by, str):
            group_by = [group_by]
        
        expected_subset = expected[[target_col] + group_by].copy()
        actual_subset = actual[[target_col] + group_by].copy()
        
        expected_subset = expected_subset.copy()
        actual_subset = actual_subset.copy()
        
        expected_subset['_dataset'] = 'expected'
        actual_subset['_dataset'] = 'actual'
        
        combined = pd.concat([expected_subset, actual_subset], ignore_index=True)
        
        results = {}
        details_dict = {}
        
        for group, group_data in combined.groupby(group_by):
            expected_group = group_data[group_data['_dataset'] == 'expected'].drop('_dataset', axis=1)
            actual_group = group_data[group_data['_dataset'] == 'actual'].drop('_dataset', axis=1)
            
            if expected_group.shape[0] == 0 or actual_group.shape[0] == 0:
                results[group] = 999999
                continue
            
            if return_details:
                psi_value, detail = _calculate_single_psi(
                    expected_group[target_col], 
                    actual_group[target_col], 
                    buckets, 
                    equal_freq,
                    True,
                    min_bin_prop,
                    content,
                    precision,
                    missing_policy,
                    psi_missing_bucket_policy,
                )
                results[group] = psi_value
                details_dict[group] = detail
            else:
                results[group] = _calculate_single_psi(
                    expected_group[target_col], 
                    actual_group[target_col], 
                    buckets, 
                    equal_freq,
                    False,
                    min_bin_prop,
                    content,
                    precision,
                    missing_policy,
                    psi_missing_bucket_policy,
                )
        
        if return_details:
            return results, details_dict
        else:
            return pd.DataFrame(results, index=['psi']).T
    
    else:
        if return_details:
            return _calculate_single_psi(
                expected[target_col], 
                actual[target_col], 
                buckets, 
                equal_freq, 
                True, 
                min_bin_prop, 
                content, 
                precision,
                missing_policy,
                psi_missing_bucket_policy,
            )
        else:
            return _calculate_single_psi(
                expected[target_col], 
                actual[target_col], 
                buckets, 
                equal_freq, 
                False, 
                min_bin_prop, 
                content, 
                precision,
                missing_policy,
                psi_missing_bucket_policy,
            )


def calculate_within_psi(
    data: pd.DataFrame,
    grp_name: str,
    target_col: str,
    benchmark: Optional[Any] = None,
    equal_freq: bool = True,
    buckets: int = 10,
    return_details: bool = False,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    benchmark_display_name: Optional[str] = None,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> Union[pd.DataFrame, Dict]:
    """
    Calculate PSI values within a single dataset, comparing groups to a benchmark.
    
    This function computes PSI between a benchmark group and all other groups
    in a specified column, useful for monitoring population stability over time.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset containing all groups.
    grp_name : str
        Column name for grouping.
    target_col : str
        Column name to calculate PSI for.
    benchmark : str or callable, optional
        Benchmark group value or filter function. Default is None: every group is compared with the whole dataset
        (the code does not pick the first group). A callable receives ``data`` and must return a boolean mask that
        selects the benchmark rows; every group is then compared with those rows.
    equal_freq : bool, optional
        Use equal frequency binning. Default is True.
    buckets : int, optional
        Number of bins. Default is 10.
    return_details : bool, optional
        Return detailed bin information. Default is False.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    benchmark_display_name : str, optional
        Custom name for benchmark in results. Default is None. When given, the result gets an extra first row
        with this name and a PSI of 0; otherwise the benchmark group is not listed.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
        naming the NaN counts.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace": when such a bin
        exists, one is added to every bin count before the shares are computed. "floor_1e6" clips the bin shares at
        ``content``, so the empty side counts as ``content``. "exclude" drops the one-sided bins from the sum. Any
        other value raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame or dict
        If return_details is False: DataFrame with a column named as the ``grp_name`` argument (the group value)
        and the column ``psi``, one row per group other than the benchmark.
        If return_details is True: Dict with 'psi' (that DataFrame) and 'details' (dict ``{group value: details
        DataFrame}``, see ``calculate_psi``).

    Examples
    --------
    >>> # Compare all months against January
    >>> psi_results = calculate_within_psi(data, 'month', 'score', benchmark='2024-01')
    """
    if callable(benchmark):
        benchmark_data = data[benchmark(data)]
    else:
        benchmark_data = data[data[grp_name] == benchmark] if benchmark is not None else data
    
    obs_values = [x for x in data[grp_name].unique().tolist() if x != benchmark]
    
    res_dict = {benchmark_display_name: 0} if benchmark_display_name is not None else {}
    detail_dict = {}
    
    for obs_value in obs_values:
        obs_data = data[data[grp_name] == obs_value]
        
        if return_details:
            psi, details = calculate_psi(
                benchmark_data, 
                obs_data, 
                target_col=target_col, 
                buckets=buckets,
                equal_freq=equal_freq, 
                return_details=True, 
                min_bin_prop=min_bin_prop, 
                content=content, 
                precision=precision,
                missing_policy=missing_policy,
                psi_missing_bucket_policy=psi_missing_bucket_policy,
            )
            res_dict[obs_value] = psi
            detail_dict[obs_value] = details
        else:
            psi = calculate_psi(
                benchmark_data, 
                obs_data, 
                target_col=target_col, 
                buckets=buckets,
                equal_freq=equal_freq, 
                return_details=False, 
                min_bin_prop=min_bin_prop, 
                content=content, 
                precision=precision,
                missing_policy=missing_policy,
                psi_missing_bucket_policy=psi_missing_bucket_policy,
            )
            res_dict[obs_value] = psi
    
    fnl_res = pd.DataFrame(res_dict, index=['psi']).T\
        .reset_index(drop=False)\
        .rename(columns={"index": grp_name})
    
    if return_details:
        return {"psi": fnl_res, "details": detail_dict}
    else:
        return fnl_res


def calculate_psi_within_dataset(
    data: pd.DataFrame,
    grp_name: str,
    varlist: List[str],
    benchmark: Optional[Any] = None,
    equal_freq: bool = True,
    buckets: int = 10,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> pd.DataFrame:
    """
    Calculate PSI for multiple variables within a dataset, comparing groups to a benchmark.
    
    This function iterates over a list of variables and calculates PSI for each,
    combining results into a single DataFrame.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.
    grp_name : str
        Column name for grouping.
    varlist : list
        List of variable names to calculate PSI for.
    benchmark : str or callable, optional
        Benchmark group value or filter function. Default is None: every group is compared with the whole dataset.
        A callable receives ``data`` and must return a boolean mask that selects the benchmark rows. The benchmark
        group itself is not listed in the result.
    equal_freq : bool, optional
        Use equal frequency binning. Default is True.
    buckets : int, optional
        Number of bins. Default is 10.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
        naming the NaN counts.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace": when such a bin
        exists, one is added to every bin count before the shares are computed. "floor_1e6" clips the bin shares at
        ``content``, so the empty side counts as ``content``. "exclude" drops the one-sided bins from the sum. Any
        other value raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        Combined PSI results for all variables, sorted by group: a column named as the ``grp_name`` argument, the
        column ``psi`` and the column ``var``. The rows of each variable are sorted by group.
        
    Examples
    --------
    >>> variables = ['score', 'age', 'income']
    >>> psi_df = calculate_psi_within_dataset(data, 'month', variables, benchmark='2024-01')
    """
    fnl_psi_res = []
    for var in tqdm(varlist):
        single_psi = calculate_within_psi(
            data=data, 
            grp_name=grp_name, 
            benchmark=benchmark, 
            target_col=var, 
            buckets=buckets, 
            equal_freq=equal_freq, 
            return_details=False, 
            min_bin_prop=min_bin_prop, 
            content=content, 
            precision=precision,
            missing_policy=missing_policy,
            psi_missing_bucket_policy=psi_missing_bucket_policy,
        ).sort_values([grp_name]).reset_index(drop=True)
        
        single_psi['var'] = var
        fnl_psi_res.append(single_psi)
    
    return pd.concat(fnl_psi_res)


def calculate_multivar_psi_two_sets(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    varlist: List[str],
    group_by: Optional[Union[str, List[str]]] = None,
    buckets: int = 10,
    equal_freq: bool = True,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> pd.DataFrame:
    """
    Calculate PSI for multiple variables by comparing two different datasets.
    
    This function computes PSI for each variable in varlist between expected
    and actual DataFrames.
    
    Parameters
    ----------
    expected_df : pandas.DataFrame
        Expected/baseline dataset.
    actual_df : pandas.DataFrame
        Actual/comparison dataset.
    varlist : list
        List of variable names to calculate PSI for.
    group_by : str or list, optional
        Column(s) to group by. Default is None.
    buckets : int, optional
        Number of bins. Default is 10.
    equal_freq : bool, optional
        Use equal frequency binning. Default is True.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
        naming the NaN counts.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace": when such a bin
        exists, one is added to every bin count before the shares are computed. "floor_1e6" clips the bin shares at
        ``content``, so the empty side counts as ``content``. "exclude" drops the one-sided bins from the sum. Any
        other value raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame
        PSI results for all variables with 'var' and 'psi' columns. With ``group_by``, the group values are in the
        index (one row per variable and group).

    Examples
    --------
    >>> variables = ['score', 'age', 'income']
    >>> psi_df = calculate_multivar_psi_two_sets(train_df, production_df, variables)
    """
    multi_psi_res = []
    for var in tqdm(varlist):
        single_psi = calculate_psi(
            expected=expected_df, 
            actual=actual_df, 
            target_col=var, 
            group_by=group_by, 
            buckets=buckets,
            equal_freq=equal_freq,
            min_bin_prop=min_bin_prop,
            content=content,
            precision=precision,
            return_details=False,
            missing_policy=missing_policy,
            psi_missing_bucket_policy=psi_missing_bucket_policy,
        )
        if group_by is None:
            single_psi = pd.DataFrame([single_psi], columns=['psi'])
        single_psi['var'] = var
        multi_psi_res.append(single_psi)
    
    return pd.concat(multi_psi_res)


def _coerce_psi_detail_frame(
    detail_obj,
    group_by: Optional[Union[str, List[str]]] = None,
) -> Optional[pd.DataFrame]:
    """Normalize PSI detail payloads before DataFrame-specific handling."""
    if detail_obj is None:
        return None
    if isinstance(detail_obj, pd.DataFrame):
        return detail_obj.copy()
    if isinstance(detail_obj, dict):
        frames = []
        group_cols = [group_by] if isinstance(group_by, str) else list(group_by or [])
        for key, value in detail_obj.items():
            if value is None:
                continue
            frame = value.copy() if isinstance(value, pd.DataFrame) else pd.DataFrame(value)
            if frame.empty:
                continue
            key_values = key if isinstance(key, tuple) else (key,)
            for col, val in zip(group_cols, key_values):
                frame[col] = val
            frames.append(frame)
        return pd.concat(frames, ignore_index=False) if frames else pd.DataFrame()
    return pd.DataFrame(detail_obj)


# def calculate_multigroup_psi_two_sets(
#     expected_df: pd.DataFrame,
#     actual_df: pd.DataFrame,
#     varlist: List[str],
#     group_by: Optional[Union[str, List[str]]] = None,
#     buckets: int = 10,
#     equal_freq: bool = True,
#     min_bin_prop: float = 0.05,
#     content: float = 1e-6,
#     precision: int = 5,
#     group_name: Optional[str] = None
# ) -> pd.DataFrame:
#     """
#     Calculate grouped PSI using expected DataFrame as benchmark, applied to actual DataFrame groups.
    
#     This function uses expected_df as the baseline and calculates PSI for each group
#     in actual_df, useful for comparing multiple time periods or segments.
    
#     Parameters
#     ----------
#     expected_df : pandas.DataFrame
#         Expected/baseline dataset used as benchmark.
#     actual_df : pandas.DataFrame
#         Actual/comparison dataset to iterate over groups.
#     varlist : list
#         List of variable names to calculate PSI for.
#     group_by : str or list, optional
#         Column(s) to group by. Default is None.
#     buckets : int, optional
#         Number of bins. Default is 10.
#     equal_freq : bool, optional
#         Use equal frequency binning. Default is True.
#     min_bin_prop : float, optional
#         Minimum proportion for each bin. Default is 0.05.
#     content : float, optional
#         Small value to avoid division by zero. Default is 1e-6.
#     precision : int, optional
#         Decimal precision. Default is 5.
#     group_name : str, optional
#         Column name for grouping in actual_df. If provided, iterates over groups.
        
#     Returns
#     -------
#     pandas.DataFrame
#         Grouped PSI results for all variables and groups.
        
#     Examples
#     --------
#     >>> # Calculate PSI for each month against Q1 benchmark
#     >>> psi_df = calculate_multigroup_psi_two_sets(
#     ...     expected_df=q1_df, 
#     ...     actual_df=all_months_df, 
#     ...     varlist=['score', 'age'],
#     ...     group_name='month'
#     ... )
#     """
#     if group_name is not None:
#         if actual_df[group_name].isna().sum() > 0:
#             actual_df = actual_df.copy()
#             actual_df[group_name] = actual_df[group_name].fillna('__NULL__')
        
#         multi_psi_res = []
#         for group, group_data in actual_df.groupby(group_name):
#             group_psi = calculate_multivar_psi_two_sets(
#                 expected_df=expected_df, 
#                 actual_df=group_data, 
#                 varlist=varlist, 
#                 group_by=None, 
#                 buckets=buckets, 
#                 equal_freq=equal_freq, 
#                 min_bin_prop=min_bin_prop, 
#                 content=content, 
#                 precision=precision
#             )
#             group_psi[group_name] = group
#             multi_psi_res.append(group_psi)
        
#         return pd.concat(multi_psi_res)
    
#     group_psi = calculate_multivar_psi_two_sets(
#         expected_df=expected_df, 
#         actual_df=actual_df, 
#         varlist=varlist, 
#         group_by=None, 
#         buckets=buckets, 
#         equal_freq=equal_freq, 
#         min_bin_prop=min_bin_prop, 
#         content=content, 
#         precision=precision
#     )
    
#     return group_psi

def calculate_multigroup_psi_two_sets(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    varlist: List[str],
    group_by: Optional[Union[str, List[str]]] = None,
    buckets: int = 10,
    equal_freq: bool = True,
    min_bin_prop: float = 0.05,
    content: float = 1e-6,
    precision: int = 5,
    group_name: Optional[str] = None,
    return_details: bool = False,
    missing_policy: str = "include",
    psi_missing_bucket_policy: str = "smooth_laplace",
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Calculate grouped PSI using expected DataFrame as benchmark, applied to actual DataFrame groups.
    
    Parameters
    ----------
    expected_df : pandas.DataFrame
        Expected/baseline dataset.
    actual_df : pandas.DataFrame
        Actual/comparison dataset.
    varlist : list
        List of variable names to calculate PSI for.
    group_by : str or list, optional
        Group column name(s); PSI is computed separately for each group. Default is None. Legacy argument: when
        ``group_name`` is None and ``return_details`` is False it is ignored (use ``group_name`` for PSI by group).
    buckets : int, optional
        Number of bins. Default is 10.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    min_bin_prop : float, optional
        Minimum proportion for each bin. Default is 0.05.
    content : float, optional
        Small value to avoid division by zero. Default is 1e-6.
    precision : int, optional
        Decimal precision. Default is 5.
    group_name : str, optional
        Name of the group column for multi-group calculation. Default is None. When it is given (and ``group_by``
        is None), every value of this column of ``actual_df`` (NaN becomes "__NULL__") is compared with the whole
        ``expected_df``, and each variable is binned once on ``expected_df``.
    return_details : bool, optional
        Whether to return detailed bin information. Default is False. If True, return a dict
        {'psi': psi_df, 'details': details_df};
        details_df contains the columns: ['bin', 'expected_count', 'actual_count', 'expected_percent',
        'actual_percent', 'psi_component', 'bucket_status', 'psi_missing_bucket_policy', group_name, 'var']
        (without ``group_name`` when it is not given; with ``group_name`` and no ``group_by`` it also holds
        'psi_component_floor_1e6').
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward compatibility). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" to keep the legacy numbers while emitting
        a RuntimeWarning that reports the NaN counts on both sides.
    psi_missing_bucket_policy : {"smooth_laplace", "floor_1e6", "exclude"}, optional
        How a bin that holds rows on one side only is treated. Default is "smooth_laplace": when such a bin
        exists, one is added to every bin count before the shares are computed. "floor_1e6" clips the bin shares at
        ``content``, so the empty side counts as ``content``. "exclude" drops the one-sided bins from the sum. Any
        other value raises ``ValueError``.

    Returns
    -------
    pandas.DataFrame or dict
        A DataFrame with the columns ``var`` and ``psi`` (plus the ``group_name`` column when it is given), or, if
        ``return_details`` is True, the dict ``{'psi': DataFrame, 'details': DataFrame}``.
    """
    if group_name is not None and group_by is None:
        return _calculate_grouped_psi_fixed_reference(
            expected_df=expected_df,
            actual_df=actual_df,
            varlist=varlist,
            group_name=group_name,
            buckets=buckets,
            equal_freq=equal_freq,
            min_bin_prop=min_bin_prop,
            content=content,
            precision=precision,
            return_details=return_details,
            missing_policy=missing_policy,
            psi_missing_bucket_policy=psi_missing_bucket_policy,
        )

    if group_name is not None:
        if actual_df[group_name].isna().sum() > 0:
            actual_df = actual_df.copy()
            actual_df[group_name] = actual_df[group_name].fillna('__NULL__')
        
        if return_details:
            psi_records = []
            detail_records = []
            
            for group, group_data in actual_df.groupby(group_name):
                for var in tqdm(varlist, desc=f"Processing group {group}"):
                    psi_val, detail_df = calculate_psi(
                        expected=expected_df,
                        actual=group_data,
                        target_col=var,
                        group_by=group_by,
                        buckets=buckets,
                        equal_freq=equal_freq,
                        return_details=True,
                        min_bin_prop=min_bin_prop,
                        content=content,
                        precision=precision,
                        missing_policy=missing_policy,
                        psi_missing_bucket_policy=psi_missing_bucket_policy,
                    )
                    
                    psi_records.append({group_name: group, 'var': var, 'psi': psi_val})
                    
                    # ========== Normalize detail_df (core of the fix) ==========
                    detail_df = _coerce_psi_detail_frame(detail_df, group_by=group_by)
                    if detail_df is not None and not detail_df.empty:
                        if 'bin' not in detail_df.columns:
                            # Reset the index; the original index column may be named 'index' or something else
                            detail_df = detail_df.reset_index()
                            # After reset_index, the first column is the original index column
                            index_col = detail_df.columns[0]
                            detail_df = detail_df.rename(columns={index_col: 'bin'})
                        
                        detail_df[group_name] = group
                        detail_df['var'] = var
                        
                        required_cols = ['bin', 'expected_count', 'actual_count', 'expected_percent', 'actual_percent', 'psi_component', 'bucket_status', 'psi_missing_bucket_policy', group_name, 'var']
                        for col in required_cols:
                            if col not in detail_df.columns:
                                detail_df[col] = np.nan
                        detail_df = detail_df[required_cols]
                        detail_records.append(detail_df)
                    # ======================================================
            
            psi_df = pd.DataFrame(psi_records)
            details_df = pd.concat(detail_records, ignore_index=True) if detail_records else pd.DataFrame(columns=['bin', 'expected_count', 'actual_count', 'expected_percent', 'actual_percent', 'psi_component', group_name, 'var'])
            return {'psi': psi_df, 'details': details_df}

        multi_psi_res = []
        for group, group_data in actual_df.groupby(group_name, dropna=False):
            group_psi = calculate_multivar_psi_two_sets(
                expected_df=expected_df,
                actual_df=group_data,
                varlist=varlist,
                group_by=group_by,
                buckets=buckets,
                equal_freq=equal_freq,
                min_bin_prop=min_bin_prop,
                content=content,
                precision=precision,
                missing_policy=missing_policy,
                psi_missing_bucket_policy=psi_missing_bucket_policy,
            )
            group_psi[group_name] = group
            multi_psi_res.append(group_psi)
        return pd.concat(multi_psi_res, ignore_index=True) if multi_psi_res else pd.DataFrame(
            columns=["var", "psi", group_name]
        )
        
    else:
        # When group_name is not specified
        if return_details:
            # Supports multiple variables
            psi_records = []
            detail_records = []
            for var in tqdm(varlist, desc="Calculating PSI with details"):
                psi_val, detail_df = calculate_psi(
                    expected=expected_df,
                    actual=actual_df,
                    target_col=var,
                    group_by=group_by,
                    buckets=buckets,
                    equal_freq=equal_freq,
                    return_details=True,
                    min_bin_prop=min_bin_prop,
                    content=content,
                    precision=precision,
                    missing_policy=missing_policy,
                    psi_missing_bucket_policy=psi_missing_bucket_policy,
                )
                psi_records.append({'var': var, 'psi': psi_val})
                
                # Normalize detail_df
                detail_df = _coerce_psi_detail_frame(detail_df, group_by=group_by)
                if detail_df is not None and not detail_df.empty:
                    if 'bin' not in detail_df.columns:
                        detail_df = detail_df.reset_index()
                        index_col = detail_df.columns[0]
                        detail_df = detail_df.rename(columns={index_col: 'bin'})
                    detail_df['var'] = var
                    required_cols = ['bin', 'expected_count', 'actual_count', 'expected_percent', 'actual_percent', 'psi_component', 'bucket_status', 'psi_missing_bucket_policy', 'var']
                    for col in required_cols:
                        if col not in detail_df.columns:
                            detail_df[col] = np.nan
                    detail_df = detail_df[required_cols]
                    detail_records.append(detail_df)
            
            psi_df = pd.DataFrame(psi_records)
            details_df = pd.concat(detail_records, ignore_index=True) if detail_records else pd.DataFrame(columns=['bin', 'expected_count', 'actual_count', 'expected_percent', 'actual_percent', 'psi_component', 'var'])
            return {'psi': psi_df, 'details': details_df}
        
        else:
            
            # Details not requested: use the original batch calculation function
            group_psi = calculate_multivar_psi_two_sets(
                expected_df=expected_df, 
                actual_df=actual_df, 
                varlist=varlist, 
                group_by=None, 
                buckets=buckets, 
                equal_freq=equal_freq, 
                min_bin_prop=min_bin_prop, 
                content=content, 
                precision=precision,
                missing_policy=missing_policy,
                psi_missing_bucket_policy=psi_missing_bucket_policy,
            )
            
            return group_psi


def _calculate_grouped_psi_fixed_reference(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    varlist: List[str],
    group_name: str,
    buckets: int,
    equal_freq: bool,
    min_bin_prop: float,
    content: float,
    precision: int,
    return_details: bool,
    missing_policy: str,
    psi_missing_bucket_policy: str,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Calculate grouped PSI after binning each variable only once."""
    if missing_policy not in {"include", "drop", "warn_and_drop"}:
        raise ValueError(
            "calculate_multigroup_psi_two_sets: missing_policy must be one of "
            f"'include', 'drop', 'warn_and_drop'; got {missing_policy!r}."
        )
    _validate_psi_bucket_policy(
        psi_missing_bucket_policy,
        "calculate_multigroup_psi_two_sets",
    )
    if group_name not in actual_df.columns:
        raise KeyError(group_name)

    group_values = actual_df[group_name].where(
        actual_df[group_name].notna(),
        "__NULL__",
    )
    group_codes, group_levels = pd.factorize(group_values, sort=True)
    n_groups = len(group_levels)
    group_sizes = np.bincount(group_codes, minlength=n_groups).astype(float)
    psi_records = []
    detail_records = []

    for var in tqdm(varlist, desc=f"Calculating PSI by {group_name}"):
        expected_series = expected_df[var].reset_index(drop=True)
        actual_series = actual_df[var].reset_index(drop=True)
        expected_valid = expected_series.notna().to_numpy()
        actual_valid = actual_series.notna().to_numpy()
        n_expected_na = int((~expected_valid).sum())
        n_actual_na = int((~actual_valid).sum())

        if missing_policy == "warn_and_drop" and (n_expected_na or n_actual_na):
            warnings.warn(
                f"calculate_multigroup_psi_two_sets: dropping NaN rows for {var!r}. "
                f"expected: {n_expected_na}/{len(expected_series)} NaN, "
                f"actual: {n_actual_na}/{len(actual_series)} NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

        expected_clean = expected_series[expected_valid]
        actual_clean = actual_series[actual_valid]
        expected_bins, breakpoints = quick_binning(
            pd.DataFrame({var: expected_clean}),
            var,
            labels=None,
            nbins=buckets,
            precision=precision,
            equal_freq=equal_freq,
            right=True,
            include_lowest=False,
            min_bin_prop=min_bin_prop,
            tree_binning=False,
            target=None,
            random_state=42,
        )
        actual_bins, _ = quick_binning(
            pd.DataFrame({var: actual_clean}),
            var,
            labels=None,
            nbins=list(breakpoints),
            precision=precision,
            equal_freq=equal_freq,
            right=True,
            include_lowest=False,
            min_bin_prop=min_bin_prop,
            tree_binning=False,
            target=None,
            random_state=42,
        )

        expected_values = np.empty(len(expected_series), dtype=object)
        actual_values = np.empty(len(actual_series), dtype=object)
        expected_values[:] = None
        actual_values[:] = None
        expected_values[expected_valid] = expected_bins.astype(object).to_numpy()
        actual_values[actual_valid] = actual_bins.astype(object).to_numpy()
        if missing_policy == "include":
            expected_values[~expected_valid] = _MISSING_BIN
            actual_values[~actual_valid] = _MISSING_BIN

        expected_values = expected_values[
            np.ones(len(expected_values), dtype=bool)
            if missing_policy == "include"
            else expected_valid
        ]
        actual_selected = (
            actual_values[actual_valid]
            if missing_policy != "include"
            else actual_values
        )
        # ``value_counts`` on categorical pd.cut output keeps unused interval
        # categories. Preserve that public PSI smoothing universe even when a
        # bin has zero observations on both sides.
        bin_universe = []
        for binned in (expected_bins, actual_bins):
            if isinstance(binned.dtype, pd.CategoricalDtype):
                bin_universe.extend(binned.cat.categories.tolist())
        universe_values = np.asarray(bin_universe, dtype=object)
        joined_values = np.concatenate(
            [universe_values, expected_values, actual_selected]
        )
        bin_codes, bin_levels = pd.factorize(joined_values, sort=False)
        universe_len = len(universe_values)
        expected_codes = bin_codes[
            universe_len : universe_len + len(expected_values)
        ]
        actual_codes = np.full(len(actual_series), -1, dtype=np.int64)
        actual_codes[actual_valid if missing_policy != "include" else np.ones(len(actual_series), dtype=bool)] = (
            bin_codes[universe_len + len(expected_values) :]
        )
        n_bins = len(bin_levels)

        expected_counts_values = np.bincount(
            expected_codes[expected_codes >= 0],
            minlength=n_bins,
        ).astype(float)
        valid_actual_codes = actual_codes >= 0
        grouped_counts = np.bincount(
            group_codes[valid_actual_codes] * n_bins + actual_codes[valid_actual_codes],
            minlength=n_groups * n_bins,
        ).reshape(n_groups, n_bins).astype(float)
        expected_count = pd.Series(expected_counts_values, index=bin_levels)
        if missing_policy == "include":
            expected_denom = float(len(expected_series)) or 1.0
            actual_denoms = group_sizes
        else:
            expected_denom = float(expected_valid.sum()) or 1.0
            actual_denoms = np.bincount(
                group_codes[actual_valid],
                minlength=n_groups,
            ).astype(float)

        for group_idx, group_value in enumerate(group_levels):
            actual_count = pd.Series(grouped_counts[group_idx], index=bin_levels)
            expected_pct, actual_pct, psi_values, psi_floor = _psi_distributions_from_counts(
                expected_count,
                actual_count,
                expected_denom,
                float(actual_denoms[group_idx]) or 1.0,
                content=content,
                policy=psi_missing_bucket_policy,
            )
            psi_records.append(
                {
                    group_name: group_value,
                    "var": var,
                    "psi": float(psi_values.sum()),
                }
            )
            if return_details:
                details = pd.DataFrame(
                    {
                        "bin": bin_levels,
                        "expected_count": expected_count.to_numpy(),
                        "actual_count": actual_count.to_numpy(),
                        "expected_percent": expected_pct.to_numpy(),
                        "actual_percent": actual_pct.to_numpy(),
                        "psi_component": psi_values.to_numpy(),
                        "psi_component_floor_1e6": psi_floor.to_numpy(),
                    }
                )
                details["bucket_status"] = "common"
                details.loc[
                    (details["expected_count"] == 0) & (details["actual_count"] > 0),
                    "bucket_status",
                ] = "actual_only"
                details.loc[
                    (details["expected_count"] > 0) & (details["actual_count"] == 0),
                    "bucket_status",
                ] = "expected_only"
                details["psi_missing_bucket_policy"] = psi_missing_bucket_policy
                details[group_name] = group_value
                details["var"] = var
                detail_records.append(details)

    psi_df = pd.DataFrame(psi_records, columns=[group_name, "var", "psi"])
    if not return_details:
        return psi_df
    details_df = pd.concat(detail_records, ignore_index=True) if detail_records else pd.DataFrame(
        columns=[
            "bin", "expected_count", "actual_count", "expected_percent",
            "actual_percent", "psi_component", "psi_component_floor_1e6",
            "bucket_status", "psi_missing_bucket_policy", group_name, "var",
        ]
    )
    return {"psi": psi_df, "details": details_df}
