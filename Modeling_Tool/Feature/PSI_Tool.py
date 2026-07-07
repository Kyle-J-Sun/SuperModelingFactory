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
        """
        if missing_policy not in {"include", "drop", "warn_and_drop"}:
            raise ValueError(
                f"PSICalculator.__init__: missing_policy must be one of "
                f"'include', 'drop', 'warn_and_drop'; got {missing_policy!r}."
            )
        _validate_psi_bucket_policy(psi_missing_bucket_policy, "PSICalculator.__init__")
        self.buckets = buckets
        self.equal_freq = equal_freq
        self.min_bin_prop = min_bin_prop
        self.content = content
        self.precision = precision
        self.missing_policy = missing_policy
        self.psi_missing_bucket_policy = psi_missing_bucket_policy
    
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
            Column to group by in both datasets.
        group_name : str, optional
            Specific group column name for multi-group calculation.
        missing_policy : {"drop", "include", "warn_and_drop"}, optional
            How NaN rows are handled. If None (default), falls back to the value
            configured on ``self.missing_policy``. Pass an explicit value to
            override the class-level default for this call only. See
            ``PSICalculator.__init__`` for the semantics of each mode.
            
        Returns
        -------
        pandas.DataFrame
            Grouped PSI results.
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
        
    Returns
    -------
    float, pandas.DataFrame, or tuple
        - If group_by is None and return_details is False: Single PSI float value.
        - If group_by is None and return_details is True: Tuple of (psi_value, details_dict).
        - If group_by is set and return_details is False: DataFrame with PSI values per group.
        - If group_by is set and return_details is True: Tuple of (results_dict, details_dict).
        
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
        Benchmark group value or filter function. If None, uses the first group.
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
        Custom name for benchmark in results.
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        How NaN rows are handled. Default in 0.5.0 is "include", which routes
        NaN rows through a dedicated "__MISSING__" bin so missing-rate drift
        contributes to the PSI. This is the recommended production behaviour
        and became the default in 0.5.0 (previously "drop" in 0.4.2 for
        numeric backward-compat). Pass "drop" to reproduce pre-0.5.0
        numbers, or "warn_and_drop" for legacy numbers with a RuntimeWarning
        naming the NaN counts.
        
    Returns
    -------
    pandas.DataFrame or dict
        If return_details is False: DataFrame with PSI values per group.
        If return_details is True: Dict with 'psi' (DataFrame) and 'details' (dict).
        
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
        Benchmark group value or filter function.
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
        
    Returns
    -------
    pandas.DataFrame
        Combined PSI results for all variables, sorted by group.
        
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
        
    Returns
    -------
    pandas.DataFrame
        PSI results for all variables with 'var' and 'psi' columns.
        
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
        基准/期望分布数据集。
    actual_df : pandas.DataFrame
        实际/对比分布数据集。
    varlist : list
        待计算 PSI 的变量名列表。
    group_by : str or list, optional
        分组列名，对每个分组分别计算 PSI。默认 None。
    buckets : int, optional
        分箱数，默认 10。
    equal_freq : bool, optional
        是否等频分箱，默认 True。
    min_bin_prop : float, optional
        每箱最小占比，默认 0.05。
    content : float, optional
        防除零小量，默认 1e-6。
    precision : int, optional
        数值精度，默认 5。
    group_name : str, optional
        多分组计算时的分组列名。默认 None。
    return_details : bool, optional
        是否返回详细分箱信息。若为 True，返回字典 {'psi': psi_df, 'details': details_df}，
        details_df 包含列：['bin', 'expected_percent', 'actual_percent', 'psi_component', group_name, 'var']
    missing_policy : {"drop", "include", "warn_and_drop"}, optional
        NaN 行处理策略。0.5.0 起默认为 "include"，将 NaN 行汇入独立的
        "__MISSING__" 桶，使缺失率漂移进入 PSI 计算；这是推荐的生产行为，
        并在 0.5.0 成为默认值（0.4.2 为 "drop" 以保持数值向后兼容）。
        传入 "drop" 可复现 0.5.0 之前的数值，或 "warn_and_drop" 保留旧数值
        同时发出 RuntimeWarning 报告两侧 NaN 数量。
    """
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
                    
                    # ========== 标准化 detail_df（修复后的核心代码） ==========
                    detail_df = _coerce_psi_detail_frame(detail_df, group_by=group_by)
                    if detail_df is not None and not detail_df.empty:
                        if 'bin' not in detail_df.columns:
                            # 重置索引，原索引列可能名为 'index' 或其他
                            detail_df = detail_df.reset_index()
                            # reset_index 后第一列就是原来的索引列
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
        
    else:
        # 未指定 group_name 时
        if return_details:
            # 支持多个变量
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
                
                # 标准化 detail_df
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
            
            # 不返回详情，使用原有的批量计算函数
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
