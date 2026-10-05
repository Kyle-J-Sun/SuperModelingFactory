"""
WOE transformation and monotonicity analysis toolkit.
Provides WOE binning, transformation, mapping, and monotonicity checking.
"""

import numpy as np
import pandas as pd
import logging
import warnings

from Modeling_Tool.Core.Binning_Tool import (
    _parse_bin_range_bounds,
    get_bin_range_list,
    run_binning,
    super_binning,
)
from Modeling_Tool.Core.Slope_Tool import calculate_slope_manual
from Modeling_Tool.Core.utils import _WOE_PURE_BIN_EPS, _calc_woe_iv_values, calc_iv, calc_woe


def _vectorized_lookup(values, mapping_keys, mapping_values):
    """Map a whole Series through a unique lookup table with integer codes."""
    key_index = pd.Index(np.asarray(list(mapping_keys), dtype=object))
    if not key_index.is_unique:
        raise ValueError("WOE mapping keys must be unique")

    positions = key_index.get_indexer(values.astype(object).to_numpy())
    output = np.full(len(values), np.nan, dtype=float)
    valid = positions >= 0
    if valid.any():
        mapped = np.asarray(list(mapping_values), dtype=float)
        output[valid] = np.take(mapped, positions[valid])
    return pd.Series(output, index=values.index)


def _vectorized_group_slopes(data, group_col, value_col):
    """Calculate per-group OLS slopes without ``groupby.apply`` callbacks."""
    work = data[[group_col, value_col]].copy()
    work["_smf_x"] = work.groupby(group_col, sort=False).cumcount().astype(float)
    work["_smf_y"] = pd.to_numeric(work[value_col], errors="coerce")
    work["_smf_xy"] = work["_smf_x"] * work["_smf_y"]
    work["_smf_x2"] = work["_smf_x"] ** 2

    grouped = work.groupby(group_col, sort=False).agg(
        n=("_smf_y", "size"),
        n_valid=("_smf_y", "count"),
        sum_x=("_smf_x", "sum"),
        sum_y=("_smf_y", "sum"),
        sum_xy=("_smf_xy", "sum"),
        sum_x2=("_smf_x2", "sum"),
    )
    numerator = grouped["n"] * grouped["sum_xy"] - grouped["sum_x"] * grouped["sum_y"]
    denominator = grouped["n"] * grouped["sum_x2"] - grouped["sum_x"] ** 2
    slopes = numerator / denominator
    slopes[(grouped["n"] != grouped["n_valid"]) | denominator.eq(0)] = np.nan
    slopes.name = "SLOPE"
    return slopes

def is_monotonic(data, column, direction='auto', strict=False, handle_nan='drop'):
    """Check whether a pandas Series or DataFrame column is monotone (increasing or decreasing).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    column : str
        Name of the column to check.
    direction : str, optional
        Direction to check: 'auto' (detect automatically), 'increasing', or 'decreasing'.
        Default is 'auto'.
    strict : bool, optional
        Whether to require strict monotonicity (equal values are not allowed). Default is False.
    handle_nan : str, optional
        How to handle NaN values: 'drop' (ignore), 'forward' (forward fill),
        'backward' (backward fill), or 'error' (raise an error). Default is 'drop'.

    Returns
    -------
    tuple
        Tuple of (is_monotone, direction).
        is_monotone is a bool; direction is 1 (increasing), -1 (decreasing), or 0 (not monotone).
    """
    series = data[column]

    # Handle NaN values
    if series.isna().any():
        if handle_nan == 'drop':
            series = series.dropna()
        elif handle_nan == 'forward':
            series = series.ffill()
        elif handle_nan == 'backward':
            series = series.bfill()
        elif handle_nan == 'error':
            raise ValueError("Series contains NaN values")
        else:
            raise ValueError("handle_nan must be one of 'drop', 'forward', 'backward' or 'error'")

    # A series that is empty or has a single element is considered monotone
    if len(series) <= 1:
        return True, 0 if len(series) == 0 else 0

    # Compute the differences
    diffs = series.diff().iloc[1:]

    # Check monotonicity
    if direction == 'auto':
        if strict:
            if (diffs > 0).all():  # all differences are positive
                return True, 1
            elif (diffs < 0).all():  # all differences are negative
                return True, -1
            else:
                return False, 0
        else:
            if (diffs >= 0).all():  # all differences are non-negative
                return True, 1
            elif (diffs <= 0).all():  # all differences are non-positive
                return True, -1
            else:
                return False, 0
    elif direction == 'increasing':
        if strict:
            is_mono = (diffs > 0).all()
        else:
            is_mono = (diffs >= 0).all()
        return is_mono, 1 if is_mono else 0
    elif direction == 'decreasing':
        if strict:
            is_mono = (diffs < 0).all()
        else:
            is_mono = (diffs <= 0).all()
        return is_mono, -1 if is_mono else 0
    else:
        raise ValueError("direction must be one of 'auto', 'increasing' or 'decreasing'")


def check_monotonicity(data, var):
    """Check the monotonicity of the WOE values.

    Verify whether the WOE values of the given variable satisfy the monotonicity requirement,
    to assess whether the binning result is consistent with business logic.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the bin information and the WOE values.
    var : str
        Name of the variable to check.

    Returns
    -------
    tuple
        Tuple of (is_monotone, direction), in the same format as returned by the is_monotonic function.

    Examples
    --------
    >>> result = check_monotonicity(woe_df, 'age')
    """
    df_grp = data.groupby([f"_bin_num_{var}"]).agg(
        {f"_bin_num_{var}": "count", var: [min, max], f"{var}_woe": [min, max, "mean"]}
    )
    assert (df_grp[(f"{var}_woe", "min")] == df_grp[(f"{var}_woe", "max")]).all()
    woe = pd.DataFrame(df_grp.loc[df_grp[(var, "min")] != df_grp[(var, "max")], (f"{var}_woe", "min")])
    res = is_monotonic(woe, (f"{var}_woe", "min"))
    return res


class WOETransformer:
    """WOE transformer.

    Provide the complete functionality of WOE binning, transformation, and monotonicity checking,
    supporting single-variable and batch multi-variable processing, as well as WOE mapping of the
    training set and the validation set.

    Parameters
    ----------
    nbins : int, optional
        Number of bins. Default is 10.
    precision : int, optional
        Numeric precision of the WOE and IV calculations. Default is 5.
    min_bin_prop : float, optional
        Minimum proportion of samples in each bin. Default is 0.05.
    include_missing : bool, optional
        Whether to treat missing values as a separate bin. Default is False.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    fillna : int/float, optional
        Value used to fill missing values. Default is -999999.
    chi2_config : tuple, optional
        Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
    tree_binning_seed : int, optional
        Random seed for decision-tree binning. Default is None.
    spec_values : list, optional
        List of special values. Default is an empty list.
    drop_bin_info : bool, optional
        Whether to drop the intermediate bin information columns. Default is True.
    ret_woe_table : bool, optional
        Whether to return the WOE mapping table. Default is True.

    Examples
    --------
    >>> transformer = WOETransformer(nbins=10)
    >>> result = transformer.transform(df, ['var1', 'var2'], 'target')
    """

    def __init__(self, nbins=10, precision=5, min_bin_prop=0.05, include_missing=False,
                 equal_freq=True, fillna=-999999, chi2_config=None, tree_binning_seed=None,
                 spec_values=None, drop_bin_info=True, ret_woe_table=True,
                 sv_min_bin_size=0.0, sv_small_policy="keep",
                 sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
        """Initialize the WOE transformer.

        Parameters
        ----------
        nbins : int, optional
            Number of bins.
        precision : int, optional
            Numeric precision of the WOE and IV calculations.
        min_bin_prop : float, optional
            Minimum proportion of samples in each bin.
        include_missing : bool, optional
            Whether to treat missing values as a separate bin.
        equal_freq : bool, optional
            Whether to use equal-frequency binning.
        fillna : int/float, optional
            Value used to fill missing values.
        chi2_config : tuple, optional
            Chi-square binning configuration.
        tree_binning_seed : int, optional
            Random seed for decision-tree binning.
        spec_values : list, optional
            List of special values.
        drop_bin_info : bool, optional
            Whether to drop the intermediate bin information columns.
        ret_woe_table : bool, optional
            Whether to return the WOE mapping table.
        sv_min_bin_size : float, optional
            Fallback threshold for low-frequency special-value (SV) bins, as a share of all samples; 0.0 = disabled (legacy behaviour).
        sv_small_policy : str, optional
            'keep' (default) / 'neutral' / 'merge_missing': how low-frequency SV bins are handled.
        sv_woe_smoothing : str, optional
            'none' (default) / 'laplace': whether the WOE of SV bins is shrunk toward the global bad rate.
        sv_smoothing_alpha : float, optional
            Laplace smoothing strength alpha (pseudo-counts); 0.0 = numerically equivalent to the legacy WOE.
        """
        if sv_small_policy not in {"keep", "neutral", "merge_missing"}:
            raise ValueError(
                f"sv_small_policy must be one of ['keep', 'neutral', 'merge_missing']; "
                f"got {sv_small_policy!r}"
            )
        if sv_woe_smoothing not in {"none", "laplace"}:
            raise ValueError(
                f"sv_woe_smoothing must be one of ['none', 'laplace']; "
                f"got {sv_woe_smoothing!r}"
            )
        if not (0.0 <= sv_min_bin_size < 1.0):
            raise ValueError(
                f"sv_min_bin_size must be in [0.0, 1.0); got {sv_min_bin_size}"
            )
        if sv_smoothing_alpha < 0.0:
            raise ValueError(
                f"sv_smoothing_alpha must be >= 0.0; got {sv_smoothing_alpha}"
            )
        self.nbins = nbins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.equal_freq = equal_freq
        self.fillna = fillna
        self.chi2_config = chi2_config
        self.tree_binning_seed = tree_binning_seed
        self.spec_values = spec_values if spec_values is not None else []
        self.drop_bin_info = drop_bin_info
        self.ret_woe_table = ret_woe_table
        self.sv_min_bin_size = sv_min_bin_size
        self.sv_small_policy = sv_small_policy
        self.sv_woe_smoothing = sv_woe_smoothing
        self.sv_smoothing_alpha = sv_smoothing_alpha

    # G19: SV-bin governance shares MonotoneWOEBinner's eps so both engines
    # produce identical numbers for the same counts.
    _SV_EPS = 1e-6

    def _sv_row_mask(self, woe_table):
        """Identify special-value bin rows: MIN == MAX and the value is in spec_values (both conditions guard against false matches)."""
        spec_values = list(self.spec_values or [])
        if not spec_values:
            return pd.Series(False, index=woe_table.index)
        return (
            woe_table["MIN"].isin(spec_values)
            & (woe_table["MIN"] == woe_table["MAX"])
            & (woe_table["N"] > 0)
        )

    def _missing_row_label(self, woe_table, sv_mask):
        """Identify the label of the missing-value bin row; return None if there is no missing bin.

        On the fit path MIN/MAX aggregate the **raw** variable column, so the missing bin is NaN throughout;
        if the caller has already applied fillna before binning (the sentinel enters the data), then MIN == MAX == fillna.
        """
        nan_bin = woe_table["MIN"].isna() & woe_table["MAX"].isna() & (woe_table["N"] > 0)
        sentinel_bin = (
            (woe_table["MIN"] == woe_table["MAX"])
            & (woe_table["MIN"] == self.fillna)
            & (woe_table["N"] > 0)
            & ~sv_mask
        )
        candidates = woe_table.index[nan_bin | sentinel_bin]
        return candidates[0] if len(candidates) else None

    def _govern_sv_bins(self, woe_table, var):
        """Apply sv_small_policy / sv_woe_smoothing to the bin rows that correspond to spec_values.

        The semantics are strictly identical to ``MonotoneWOEBinner._compute_sv_table``: the
        low-share fallback (option 1) takes priority, and smoothing (option 2) only applies to SV
        bins whose share meets the threshold (or when policy='keep').

        The missing bin is likewise a governed SV bin (in the sv_table of MonotoneWOEBinner the
        ``[Missing]`` row carries the same weight as the other SV rows), so it is added to the set of
        governed rows; it is merely **never a merge source** (target-only: it cannot be merged into
        itself). If rows were selected by ``_sv_row_mask`` alone, the missing bin (MIN/MAX aggregate the
        raw column => NaN/NaN) would slip out of the smoothing loop, and the [Missing] WOE of the two
        engines would differ under the same parameters.
        """
        eps = self._SV_EPS
        total_bad  = float(woe_table["N_BAD"].sum())
        total_good = float(woe_table["N_GOOD"].sum())
        n_total = total_bad + total_good
        p = total_bad / (n_total + eps)

        sv_mask = self._sv_row_mask(woe_table)
        missing_label = self._missing_row_label(woe_table, sv_mask)
        governed_idx = list(woe_table.index[sv_mask])
        if missing_label is not None and missing_label not in governed_idx:
            governed_idx.append(missing_label)
        if not governed_idx:
            return woe_table
        pending_merge = []
        for idx in governed_idx:
            is_small = (
                self.sv_small_policy != "keep"
                and self.sv_min_bin_size > 0.0
                and n_total > 0
                and float(woe_table.loc[idx, "N"]) / n_total < self.sv_min_bin_size
                # The missing bin is the merge target and is never a merge source.
                and not (self.sv_small_policy == "merge_missing"
                         and idx == missing_label)
            )
            if is_small and self.sv_small_policy == "neutral":
                woe_table.loc[idx, "WOE"] = 0.0
                woe_table.loc[idx, "IV"] = 0.0
            elif is_small:
                pending_merge.append(idx)
            elif self.sv_woe_smoothing == "laplace" and self.sv_smoothing_alpha > 0.0:
                # Same formula as MonotoneWOEBinner._compute_woe_single_bin: first shrink the in-bin
                # bad_rate toward the base rate p, then convert back to equivalent bad/good counts.
                a = self.sv_smoothing_alpha
                n_bad  = float(woe_table.loc[idx, "N_BAD"])
                n_good = float(woe_table.loc[idx, "N_GOOD"])
                r = (n_bad + a * p) / (n_bad + n_good + a)
                pct_bad  = ((n_bad + n_good) * r)         / (total_bad  + eps)
                pct_good = ((n_bad + n_good) * (1.0 - r)) / (total_good + eps)
                woe = float(np.log((pct_bad + eps) / (pct_good + eps)))
                woe_table.loc[idx, "BAD_PCT_PER_BIN"] = pct_bad
                woe_table.loc[idx, "GOOD_PCT_PER_BIN"] = pct_good
                woe_table.loc[idx, "WOE"] = woe
                woe_table.loc[idx, "IV"] = (pct_bad - pct_good) * woe
        if pending_merge:
            woe_table = self._merge_sv_into_missing_master(
                woe_table, pending_merge, missing_label,
                total_bad, total_good, var,
            )
        return woe_table

    def _merge_sv_into_missing_master(self, woe_table, pending_merge, missing_label,
                                      total_bad, total_good, var):
        """Merge the bad/good counts of low-frequency SV rows into the missing-bin row and recompute the WOE.

        Corresponds one-to-one to ``MonotoneWOEBinner._merge_small_into_missing``: a merged row keeps
        its own row, but its stored WOE is rewritten to the WOE recomputed for the missing bin and its IV
        is set to 0, so the transform side (``mapping_woe`` / ``convert_single_var_woe``) needs no change.
        When there is no missing bin, fall back to neutral and emit a warning.

        N/N_BAD/N_GOOD are **transferred**, not copied (the source rows are zeroed), so the total of the N column is unchanged.
        """
        eps = self._SV_EPS
        if missing_label is None:
            for idx in pending_merge:
                warnings.warn(
                    f"sv_small_policy='merge_missing' but {var!r} has no [Missing] bin; "
                    f"falling back to 'neutral' for SV "
                    f"{woe_table.loc[idx, 'MIN']!r}.",
                    UserWarning, stacklevel=2,
                )
                woe_table.loc[idx, "WOE"] = 0.0
                woe_table.loc[idx, "IV"] = 0.0
            return woe_table
        m = missing_label
        add_bad  = float(woe_table.loc[pending_merge, "N_BAD"].sum())
        add_good = float(woe_table.loc[pending_merge, "N_GOOD"].sum())
        new_n    = int(woe_table.loc[m, "N"]) + int(woe_table.loc[pending_merge, "N"].sum())
        new_bad  = float(woe_table.loc[m, "N_BAD"])  + add_bad
        new_good = float(woe_table.loc[m, "N_GOOD"]) + add_good
        pct_bad  = new_bad  / (total_bad  + eps)
        pct_good = new_good / (total_good + eps)
        woe = float(np.log((pct_bad + eps) / (pct_good + eps)))
        woe_table.loc[m, "N"] = new_n
        woe_table.loc[m, "N_BAD"] = int(new_bad)
        woe_table.loc[m, "N_GOOD"] = int(new_good)
        woe_table.loc[m, "AVG_BAD"] = (
            new_bad / (new_bad + new_good) if (new_bad + new_good) > 0 else np.nan
        )
        woe_table.loc[m, "AVG_GOOD"] = (
            new_good / (new_bad + new_good) if (new_bad + new_good) > 0 else np.nan
        )
        woe_table.loc[m, "BAD_PCT_PER_BIN"] = pct_bad
        woe_table.loc[m, "GOOD_PCT_PER_BIN"] = pct_good
        woe_table.loc[m, "WOE"] = woe
        woe_table.loc[m, "IV"] = (pct_bad - pct_good) * woe
        woe_table.loc[pending_merge, "N"] = 0
        woe_table.loc[pending_merge, "N_BAD"] = 0
        woe_table.loc[pending_merge, "N_GOOD"] = 0
        woe_table.loc[pending_merge, "AVG_BAD"] = np.nan
        woe_table.loc[pending_merge, "AVG_GOOD"] = np.nan
        woe_table.loc[pending_merge, "LIFT"] = np.nan
        woe_table.loc[pending_merge, "BAD_PCT_PER_BIN"] = 0.0
        woe_table.loc[pending_merge, "GOOD_PCT_PER_BIN"] = 0.0
        woe_table.loc[pending_merge, "WOE"] = woe
        woe_table.loc[pending_merge, "IV"] = 0.0
        # AVG_BAD has been rewritten by the merge and LIFT depends on its mean, so recompute it over the whole table.
        woe_table["LIFT"] = woe_table["AVG_BAD"] / woe_table["AVG_BAD"].mean()
        return woe_table

    def _get_woe_table(self, binning_res, var, dep):
        """Compute the WOE table from the binning result.

        Parameters
        ----------
        binning_res : pd.DataFrame
            DataFrame containing the binning result.
        var : str
            Variable name.
        dep : str
            Name of the target variable.

        Returns
        -------
        tuple
            Tuple of (woe_table, woe_mapping_dict).
        """
        working = binning_res.copy()
        target = working[dep]
        working["_smf_target_n"] = target.notna().astype(np.int64)
        working["_smf_bad_n"] = target.eq(1).astype(np.int64)
        working["_smf_good_n"] = target.eq(0).astype(np.int64)
        woe_table = working.groupby(
            [f"_bin_num_{var}", f"_bin_range_{var}"], dropna=False
        ).agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(f"_bin_num_{var}", "count"),
            AVG_SCORE=(var, "mean"),
            TARGET_N=("_smf_target_n", "sum"),
            N_BAD=("_smf_bad_n", "sum"),
            N_GOOD=("_smf_good_n", "sum"),
        )
        woe_table["AVG_BAD"] = woe_table["N_BAD"] / woe_table["TARGET_N"]
        woe_table["AVG_GOOD"] = woe_table["N_GOOD"] / woe_table["TARGET_N"]
        woe_table = woe_table.drop(columns="TARGET_N")

        # IV/WOE Calculation
        woe_table["BAD_PCT_PER_BIN"] = woe_table["N_BAD"] / woe_table["N_BAD"].sum()
        woe_table["GOOD_PCT_PER_BIN"] = woe_table["N_GOOD"] / woe_table["N_GOOD"].sum()
        woe_table["LIFT"] = woe_table['AVG_BAD'] / woe_table['AVG_BAD'].mean()
        woe_table["WOE"], woe_table["IV"] = _calc_woe_iv_values(
            woe_table, "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", pure_bin_eps=_WOE_PURE_BIN_EPS
        )

        woe_table = woe_table.reset_index(drop=False)

        # ── G19: low-frequency SV bin governance (aligned with MonotoneWOEBinner) ──
        if self.sv_small_policy != "keep" or self.sv_woe_smoothing != "none":
            woe_table = self._govern_sv_bins(woe_table, var)

        # WOE Mapping Dictionary
        woe_mapping_dict = dict(zip(woe_table[f"_bin_range_{var}"], woe_table["WOE"]))

        return woe_table, woe_mapping_dict

    def transform_single(self, train_df, var, dep, oot_df=None, check_monotonicity_flag=False):
        """Apply the WOE transformation to a single variable.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataset.
        var : str
            Name of the variable to transform.
        dep : str
            Name of the target (dependent) variable.
        oot_df : pd.DataFrame, optional
            Validation/test dataset. Default is None.
        check_monotonicity_flag : bool, optional
            Whether to check monotonicity. Default is False.

        Returns
        -------
        tuple/list
            The training result, the validation result, and the WOE mapping table, depending on the arguments.
        """
        chi2_method = False
        if self.chi2_config:
            chi2_method = True
        else:
            self.chi2_config = (100, 0.99)

        tree_binning = False
        if self.tree_binning_seed:
            tree_binning = True

        train_res, train_edges = super_binning(
            data=train_df,
            score=var,
            dep=dep,
            nbins=self.nbins,
            precision=self.precision,
            min_bin_prop=self.min_bin_prop,
            include_missing=self.include_missing,
            equal_freq=self.equal_freq,
            chi2_method=chi2_method,
            chi2_p=self.chi2_config[1],
            init_equi_bins=self.chi2_config[0],
            fillna=self.fillna,
            spec_values=self.spec_values,
            tree_binning=tree_binning,
            random_state=self.tree_binning_seed,
            return_edges=True,
            bin_colnames=(f"_bin_num_{var}", f"_bin_range_{var}"),
            ascending=True
        )

        train_woe_table, train_woe_mapping_dict = self._get_woe_table(train_res, var, dep)

        if check_monotonicity_flag:
            if not is_monotonic(train_woe_table.query("MIN != MAX"), "WOE")[0]:
                logging.warning(f"WARNING: {var} WOE values are NOT monotonic in Train Dataset!")

        # WOE Mapping to DataFrame
        train_res[f"{var}_woe"] = _vectorized_lookup(
            train_res[f"_bin_range_{var}"],
            train_woe_mapping_dict.keys(),
            train_woe_mapping_dict.values(),
        )

        # Drop Bin Info
        if self.drop_bin_info:
            train_res = train_res.drop(columns=[f"_bin_num_{var}", f"_bin_range_{var}"])

        if oot_df is not None:
            woe_mapping_table = train_woe_table.copy()
            woe_mapping_table["BIN_RANGE"] = woe_mapping_table[f"_bin_range_{var}"]
            woe_mapping_table["BIN_NUM"] = woe_mapping_table[f"_bin_num_{var}"]
            woe_mapping_table["VAR"] = var
            oot_res = _mapping_woe_single_var(
                data=oot_df,
                var=var,
                woe_mapping_table=woe_mapping_table,
                missing_ref_value=self.fillna,
            )

        if self.ret_woe_table and oot_df is not None:
            return train_res, oot_res, train_woe_table

        if self.ret_woe_table and oot_df is None:
            return train_res, train_woe_table

        if oot_df is not None:
            return train_res, oot_res

        return train_res

    def transform(self, train_df, varlist, dep, oot_df=None, check_monotonicity_flag=False):
        """Apply the WOE transformation to multiple variables.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training dataset.
        varlist : list
            List of variable names to transform.
        dep : str
            Name of the target (dependent) variable.
        oot_df : pd.DataFrame, optional
            Validation/test dataset. Default is None.
        check_monotonicity_flag : bool, optional
            Whether to check monotonicity. Default is False.

        Returns
        -------
        tuple/list
            The result dictionary and the WOE mapping table.
            The dictionary keys are 'TRAIN' and 'OOT' (the latter when oot_df is not None).

        Examples
        --------
        >>> transformer = WOETransformer(nbins=10)
        >>> result = transformer.transform(df, ['var1', 'var2'], 'target')
        """
        train_base = train_df.copy()
        oot_base = oot_df.copy() if oot_df is not None else None
        train_outputs = []
        oot_outputs = []
        table_outputs = []

        for var in varlist:
            woe_res = self.transform_single(
                train_df=train_base,
                var=var,
                dep=dep,
                oot_df=oot_base,
                check_monotonicity_flag=check_monotonicity_flag,
            )
            if self.ret_woe_table:
                if oot_base is None:
                    train_part, table_part = woe_res
                    oot_part = None
                else:
                    train_part, oot_part, table_part = woe_res
                table_part = table_part.copy()
                table_part["VAR"] = var
                table_outputs.append(
                    table_part.rename(
                        columns={
                            f"_bin_num_{var}": "BIN_NUM",
                            f"_bin_range_{var}": "BIN_RANGE",
                        }
                    )
                )
            elif oot_base is None:
                train_part = woe_res
                oot_part = None
            else:
                train_part, oot_part = woe_res

            train_cols = [f"{var}_woe"]
            if not self.drop_bin_info:
                train_cols = [f"_bin_num_{var}", f"_bin_range_{var}", *train_cols]
            train_outputs.append(train_part[train_cols])
            if oot_part is not None:
                oot_outputs.append(oot_part[[f"{var}_woe"]])

        def _combine(base, outputs):
            if not outputs:
                return base.copy()
            output_frame = pd.concat(outputs, axis=1)
            clean_base = base.drop(columns=list(output_frame.columns), errors="ignore")
            return pd.concat([clean_base, output_frame], axis=1)

        fnl_res = {"TRAIN": _combine(train_base, train_outputs)}
        if oot_base is not None:
            fnl_res["OOT"] = _combine(oot_base, oot_outputs)

        if self.ret_woe_table:
            train_woe_table = (
                pd.concat(table_outputs, ignore_index=True)
                if table_outputs
                else pd.DataFrame()
            )
            return fnl_res, train_woe_table
        return fnl_res


def convert_single_var_woe(data, var, woe_mapping_table, missing_ref=None, ret_bin_no=False):
    """Convert raw variable values to WOE values.

    Apply the WOE transformation to the given variable using a precomputed WOE mapping table.
    Supports missing-value handling and returning the bin numbers.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    var : str
        Name of the variable to transform.
    woe_mapping_table : pd.DataFrame
        WOE mapping table, containing the columns bin_no, bin_value, woe, and n.
    missing_ref : any, optional
        Reference value for missing values. Default is None.
    ret_bin_no : bool, optional
        Whether to return the bin numbers instead of the WOE values. Default is False.

    Returns
    -------
    pd.Series/pd.Categorical
        The transformed WOE values or the bin numbers.

    Examples
    --------
    >>> woe_values = convert_single_var_woe(df, 'age', woe_table)
    """
    var_woe_mapping = woe_mapping_table.query(f"var_name == '{var}_woe'")
    var_woe_mapping = var_woe_mapping[["bin_no", "bin_value", "woe", "n"]]

    left, right = _parse_bin_range_bounds(var_woe_mapping, col="bin_value")
    unique_range = np.unique(np.concatenate([left, right])).tolist()

    var_serires = data[var]
    if missing_ref is not None:
        var_serires = var_serires.fillna(missing_ref)

    bin_no_transform = pd.cut(
        var_serires,
        bins=unique_range,
        right=False,
        labels=[x for x in range(0, len(unique_range) - 1)]
    )

    if ret_bin_no:
        return bin_no_transform

    return _vectorized_lookup(
        pd.Series(bin_no_transform.astype(object), index=data.index),
        var_woe_mapping["bin_no"],
        var_woe_mapping["woe"],
    )


class WOEMappingTransformer:
    """Transformer based on a WOE mapping table.

    Apply the WOE transformation to new data using a precomputed WOE mapping table;
    supports single-variable and batch multi-variable processing.

    Parameters
    ----------
    woe_mapping_table : pd.DataFrame
        WOE mapping table.
    missing_ref : any, optional
        Reference value for missing values. Default is None.
    ret_bin_no : bool, optional
        Whether to return the bin numbers. Default is False.
    ret_category : bool, optional
        Whether to return a categorical type. Default is False.
    rename_orig_var : bool, optional
        Whether to rename the original variable. Default is False.
    suffix : str, optional
        Suffix appended to the variable names. Default is ''.

    Examples
    --------
    >>> transformer = WOEMappingTransformer(woe_mapping_table)
    >>> result = transformer.transform(df, ['var1', 'var2'])
    """

    def __init__(self, woe_mapping_table, missing_ref=None, ret_bin_no=False,
                 ret_category=False, rename_orig_var=False, suffix=''):
        """Initialize the WOE mapping transformer.

        Parameters
        ----------
        woe_mapping_table : pd.DataFrame
            WOE mapping table.
        missing_ref : any, optional
            Reference value for missing values.
        ret_bin_no : bool, optional
            Whether to return the bin numbers.
        ret_category : bool, optional
            Whether to return a categorical type.
        rename_orig_var : bool, optional
            Whether to rename the original variable.
        suffix : str, optional
            Suffix appended to the variable names.
        """
        self.woe_mapping_table = woe_mapping_table
        self.missing_ref = missing_ref
        self.ret_bin_no = ret_bin_no
        self.ret_category = ret_category
        self.rename_orig_var = rename_orig_var
        self.suffix = suffix

    def transform_single(self, data, var):
        """Apply the WOE transformation to a single variable.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        var : str
            Name of the variable to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.
        """
        if self.rename_orig_var:
            data = data.rename(columns={var: (var + self.suffix)})
            data[var] = data[(var + self.suffix)].copy()
            data[var] = convert_single_var_woe(
                data, var, self.woe_mapping_table,
                self.missing_ref, self.ret_bin_no
            )
            if not self.ret_category:
                data[var] = data[var].astype(float)
        else:
            data[var + self.suffix] = convert_single_var_woe(
                data, var, self.woe_mapping_table,
                self.missing_ref, self.ret_bin_no
            )
            if not self.ret_category:
                data[var + self.suffix] = data[var + self.suffix].astype(float)

        return data

    def transform(self, data, varlist):
        """Apply the WOE transformation to multiple variables.

        Parameters
        ----------
        data : pd.DataFrame
            Input DataFrame.
        varlist : list
            List of variable names to transform.

        Returns
        -------
        pd.DataFrame
            The transformed DataFrame.

        Examples
        --------
        >>> transformer = WOEMappingTransformer(woe_mapping_table)
        >>> result = transformer.transform(df, ['var1', 'var2'])
        """
        output_names = [var if self.rename_orig_var else var + self.suffix for var in varlist]
        outputs = {}
        for var, output_name in zip(varlist, output_names):
            values = convert_single_var_woe(
                data,
                var,
                self.woe_mapping_table,
                self.missing_ref,
                self.ret_bin_no,
            )
            if not self.ret_category:
                values = values.astype(float)
            outputs[output_name] = np.asarray(values)

        output_frame = pd.DataFrame(outputs, index=data.index)
        if self.rename_orig_var:
            rename_map = {var: var + self.suffix for var in varlist}
            base = data.rename(columns=rename_map).copy()
        else:
            base = data.copy()

        base = base.drop(columns=[col for col in output_names if col in base.columns])
        result = pd.concat([base, output_frame], axis=1)
        if not self.rename_orig_var:
            final_order = list(data.columns) + [col for col in output_names if col not in data.columns]
            result = result.reindex(columns=final_order)
        elif not self.suffix:
            result = result.reindex(columns=list(data.columns))
        return result


def woe_transform_cdaml(data, varlist, woe_mapping_path, missing_ref=None,
                        ret_bin_no=False, ret_category=False, rename_orig_var=False, suffix=''):
    """Apply the WOE transformation using the cdaml package.

    Read the WOE mapping table from a file path or use a mapping-table DataFrame directly,
    and apply the WOE transformation to the given list of variables.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    varlist : list
        List of variable names to transform.
    woe_mapping_path : str/pd.DataFrame
        Path of the WOE mapping table file, or the mapping table DataFrame itself.
    missing_ref : any, optional
        Reference value for missing values. Default is None.
    ret_bin_no : bool, optional
        Whether to return the bin numbers. Default is False.
    ret_category : bool, optional
        Whether to return a categorical type. Default is False.
    rename_orig_var : bool, optional
        Whether to rename the original variable. Default is False.
    suffix : str, optional
        Suffix appended to the variable names. Default is ''.

    Returns
    -------
    pd.DataFrame
        The transformed DataFrame.

    Examples
    --------
    >>> result = woe_transform_cdaml(df, ['var1', 'var2'], 'woe_mapping.csv')
    """
    woe_mapping_table = pd.read_csv(woe_mapping_path) if isinstance(woe_mapping_path, str) else woe_mapping_path
    woe_mapping_table.columns = [x.lower() for x in woe_mapping_table.columns]

    transformer = WOEMappingTransformer(
        woe_mapping_table=woe_mapping_table,
        missing_ref=missing_ref,
        ret_bin_no=ret_bin_no,
        ret_category=ret_category,
        rename_orig_var=rename_orig_var,
        suffix=suffix
    )
    return transformer.transform(data, varlist)


def get_woe_table(data, var, dep, grp_name=None, nbins=10, precision=5,
                  min_bin_prop=0.05, include_missing=True, equal_freq=True,
                  fillna=-999999, chi2_config=None, tree_binning_seed=None, spec_values=None):
    """Get the WOE bin table.

    Bin the given variable and compute the statistics of each bin, such as the WOE and IV values.
    Supports grouped analysis and monotonicity checking.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    var : str
        Name of the variable to analyze.
    dep : str
        Name of the target (dependent) variable.
    grp_name : str, optional
        Name of the grouping variable. Default is None.
    nbins : int, optional
        Number of bins. Default is 10.
    precision : int, optional
        Numeric precision. Default is 5.
    min_bin_prop : float, optional
        Minimum proportion of samples in each bin. Default is 0.05.
    include_missing : bool, optional
        Whether to treat missing values as a separate bin. Default is True.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    fillna : int/float, optional
        Value used to fill missing values. Default is -999999.
    chi2_config : tuple, optional
        Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
    tree_binning_seed : int, optional
        Random seed for decision-tree binning. Default is None.
    spec_values : list, optional
        List of special values. Default is None.

    Returns
    -------
    tuple/pd.DataFrame
        A (woe_table, grp_summary, grp_woe_pvt) tuple when grp_name is not None;
        a (woe_table, is_monotonic, direction, slope) tuple when grp_name is None.

    Examples
    --------
    >>> woe_table, is_mono, direction, slope = get_woe_table(df, 'age', 'target')
    """
    from pandas.api.types import is_numeric_dtype, is_string_dtype

    if spec_values is None:
        spec_values = []

    chi2_method = False
    if chi2_config:
        chi2_method = True
    else:
        chi2_config = (100, 0.99)

    tree_binning = False
    if tree_binning_seed:
        tree_binning = True

    from Modeling_Tool.Eval.Model_Eval_Tool import get_gains_table

    gains_table = get_gains_table(
        data=data,
        score=var,
        dep=dep,
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        chi2_method=chi2_method,
        chi2_p=chi2_config[1],
        tree_binning=tree_binning,
        init_equi_bins=chi2_config[0],
        fillna=fillna,
        spec_values=spec_values,
        sync_range=True,
        grp_name=grp_name,
        retSummary=False,
        random_state=tree_binning_seed,
        ascending=True,
        withSummary=False
    )

    gains_table = gains_table.reset_index(drop=False).rename(columns={
        "_bin_num": "BIN_NUM", "_bin_range": "BIN_RANGE"
    })
    woe_cols = ["BIN_NUM", "BIN_RANGE", "MIN", "MAX", "N", "RANK_ORDER_BUMP", "WOE", "IV", "AVG_BAD"]
    if grp_name:
        woe_cols += [grp_name]
    woe_table = gains_table[woe_cols]
    avg_bad_df = woe_table.loc[woe_table["BIN_NUM"] != "Grand Summary", :]

    if include_missing:
        monoto_info = is_monotonic(avg_bad_df.iloc[1:, :], "AVG_BAD")
    else:
        monoto_info = is_monotonic(avg_bad_df, "AVG_BAD")

    dep_slope = calculate_slope_manual(avg_bad_df, "AVG_BAD")
    direction = 1 if dep_slope > 0 else -1 if dep_slope < 0 else 0

    if grp_name:
        slope_grp = _vectorized_group_slopes(gains_table, grp_name, "AVG_BAD").to_frame()
        grp_summary = gains_table.groupby([grp_name]).agg(
            N=("N", sum), IV=("IV", sum), KS=("KS_PER_BIN", max),
            BTM_LIFT=("LIFT", min), TOP_LIFT=("LIFT", max)
        ).merge(slope_grp, left_index=True, right_index=True)
        slope_values = grp_summary["SLOPE"].to_numpy()
        grp_summary['direction'] = np.select(
            [slope_values > 0, slope_values < 0], [1, -1], default=0
        )
        grp_woe_pvt = gains_table.pivot_table(
            index=["BIN_NUM", "BIN_RANGE"],
            columns=[grp_name],
            values=["WOE", 'AVG_BAD']
        )

        return woe_table, grp_summary, grp_woe_pvt

    return (woe_table, monoto_info[0], direction, dep_slope)


def plot_monotonicity_check(data, column, title=None, include_missing=True):
    """Plot a series and annotate its monotonicity.

    Create a line chart of the values of the given column
    and annotate the result of the monotonicity check on the chart.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing the data.
    column : str
        Name of the column to plot.
    title : str, optional
        Chart title. Default is None (generated automatically).
    include_missing : bool, optional
        Whether to include missing-value handling. Default is True.

    Returns
    -------
    None
        The chart is displayed directly.

    Examples
    --------
    >>> plot_monotonicity_check(df, 'woe_values')
    """
    import matplotlib.pyplot as plt

    series = data[column]

    # Check monotonicity
    if include_missing:
        is_mono, direction = is_monotonic(
            data.iloc[1:, :], "AVG_BAD", strict=True, handle_nan='drop'
        )
    else:
        is_mono, direction = is_monotonic(data, "AVG_BAD", strict=True, handle_nan='drop')

    # Create the chart
    plt.figure(figsize=(10, 6))
    plt.plot(series.index, series.values, 'bo-', linewidth=2, markersize=6)

    # Add the title and labels
    if title is None:
        title = f"Monotonicity Check: {'Strict' if is_mono else 'Not'} Monotonic {direction if is_mono else ''}"
    plt.title(title, fontsize=14)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Add the grid
    plt.grid(True, alpha=0.3)

    # Show the monotonicity information
    plt.text(
        0.02, 0.98,
        f"Strict Monotonic: {is_mono}\n Direction: {direction}",
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def woe_transform(train_df, var, dep, nbins, oot_df=None, chi2_config=None, tree_binning_seed=None,
                  precision=5, min_bin_prop=0.05, include_missing=False, equal_freq=True,
                  ascending=True, fillna=-999999, spec_values=None, drop_bin_info=True,
                  ret_woe_table=True, check_monotonicity=False,
                  sv_min_bin_size=0.0, sv_small_policy="keep",
                  sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
    """Convert a variable to WOE values.

    Bin a single variable and compute its WOE values, supporting the transformation of both the
    training set and the validation set. Implemented on top of the WOETransformer class.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    var : str
        Name of the variable to transform.
    dep : str
        Name of the target (dependent) variable.
    nbins : int
        Number of bins.
    oot_df : pd.DataFrame, optional
        Validation/test dataset. Default is None.
    chi2_config : tuple, optional
        Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
    tree_binning_seed : int, optional
        Random seed for decision-tree binning. Default is None.
    precision : int, optional
        Numeric precision. Default is 5.
    min_bin_prop : float, optional
        Minimum proportion of samples in each bin. Default is 0.05.
    include_missing : bool, optional
        Whether to treat missing values as a separate bin. Default is False.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    ascending : bool, optional
        Whether to sort in ascending order. Default is True.
    fillna : int/float, optional
        Value used to fill missing values. Default is -999999.
    spec_values : list, optional
        List of special values. Default is None.
    drop_bin_info : bool, optional
        Whether to drop the intermediate bin information columns. Default is True.
    ret_woe_table : bool, optional
        Whether to return the WOE mapping table. Default is True.
    check_monotonicity : bool, optional
        Whether to check monotonicity. Default is False.
    sv_min_bin_size : float, optional
        Fallback threshold for low-frequency special-value (SV) bins, as a share of all samples. Default is 0.0 (disabled).
    sv_small_policy : str, optional
        'keep' (default) / 'neutral' / 'merge_missing'.
    sv_woe_smoothing : str, optional
        'none' (default) / 'laplace'.
    sv_smoothing_alpha : float, optional
        Laplace smoothing strength alpha. Default is 0.0.

    Returns
    -------
    tuple/list/pd.DataFrame
        Different combinations depending on the arguments:
        - ret_woe_table=True, oot_df=None: (train_res, train_woe_table)
        - ret_woe_table=True, oot_df is not None: (train_res, oot_res, train_woe_table)
        - ret_woe_table=False, oot_df is not None: (train_res, oot_res)
        - ret_woe_table=False, oot_df=None: train_res

    Examples
    --------
    >>> train_res, woe_table = woe_transform(df, 'age', 'target', nbins=10)
    """
    if spec_values is None:
        spec_values = []

    transformer = WOETransformer(
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        fillna=fillna,
        chi2_config=chi2_config,
        tree_binning_seed=tree_binning_seed,
        spec_values=spec_values,
        drop_bin_info=drop_bin_info,
        ret_woe_table=ret_woe_table,
        sv_min_bin_size=sv_min_bin_size,
        sv_small_policy=sv_small_policy,
        sv_woe_smoothing=sv_woe_smoothing,
        sv_smoothing_alpha=sv_smoothing_alpha,
    )
    return transformer.transform_single(
        train_df=train_df,
        var=var,
        dep=dep,
        oot_df=oot_df,
        check_monotonicity_flag=check_monotonicity
    )


def woe_transformation(train_df, varlist, dep, oot_df=None, nbins=10, chi2_config=None,
                       tree_binning_seed=None, precision=5, min_bin_prop=0.05,
                       include_missing=False, equal_freq=True, fillna=-999999,
                       spec_values=None, drop_bin_info=True, ret_woe_table=True):
    """Apply the WOE transformation to a list of variables.

    Bin and transform multiple variables to WOE in batch, supporting the training set and the validation set.
    Implemented on top of the WOETransformer class.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training dataset.
    varlist : list
        List of variable names to transform.
    dep : str
        Name of the target (dependent) variable.
    oot_df : pd.DataFrame, optional
        Validation/test dataset. Default is None.
    nbins : int, optional
        Number of bins. Default is 10.
    chi2_config : tuple, optional
        Chi-square binning configuration, an (init_bins, p_value) tuple. Default is None.
    tree_binning_seed : int, optional
        Random seed for decision-tree binning. Default is None.
    precision : int, optional
        Numeric precision. Default is 5.
    min_bin_prop : float, optional
        Minimum proportion of samples in each bin. Default is 0.05.
    include_missing : bool, optional
        Whether to treat missing values as a separate bin. Default is False.
    equal_freq : bool, optional
        Whether to use equal-frequency binning. Default is True.
    fillna : int/float, optional
        Value used to fill missing values. Default is -999999.
    spec_values : list, optional
        List of special values. Default is None.
    drop_bin_info : bool, optional
        Whether to drop the intermediate bin information columns. Default is True.
    ret_woe_table : bool, optional
        Whether to return the WOE mapping table. Default is True.

    Returns
    -------
    tuple
        Tuple of (result dictionary, train_woe_table).
        The result dictionary contains the 'TRAIN' key, and the validation set is under the 'OOT' key (when oot_df is not None).

    Examples
    --------
    >>> result, woe_table = woe_transformation(df, ['var1', 'var2'], 'target')
    """
    if spec_values is None:
        spec_values = []

    transformer = WOETransformer(
        nbins=nbins,
        precision=precision,
        min_bin_prop=min_bin_prop,
        include_missing=include_missing,
        equal_freq=equal_freq,
        fillna=fillna,
        chi2_config=chi2_config,
        tree_binning_seed=tree_binning_seed,
        spec_values=spec_values,
        drop_bin_info=drop_bin_info,
        ret_woe_table=ret_woe_table
    )
    return transformer.transform(train_df, varlist, dep, oot_df)


def _map_woe_arrays(data, var, woe_mapping_table, suffix="_woe", missing_ref_value=-999999):
    """Return vectorized WOE and bin columns for one feature."""
    var_woe_mapping = woe_mapping_table.loc[
        woe_mapping_table["VAR"].eq(var), ["BIN_NUM", "BIN_RANGE", "WOE", "N"]
    ].copy()
    if var_woe_mapping.empty:
        raise KeyError(f"No WOE mapping rows found for variable {var!r}")

    first_range = var_woe_mapping["BIN_RANGE"].astype("string").dropna().iloc[0]
    include_lowest = first_range.startswith("[")
    right = first_range.endswith("]")
    bin_range = get_bin_range_list(var_woe_mapping, col="BIN_RANGE")

    binned, _ = run_binning(
        data=data[[var]].copy(),
        column=var,
        nbins=sorted(set([*bin_range, -np.inf, np.inf])),
        include_missing=True,
        equal_freq=True,
        bin_colnames=("_BIN_NUM_", "_BIN_RANGE_"),
        ascending=True,
        include_lowest=include_lowest,
        right=right,
        fillna=missing_ref_value,
    )
    woe_values = _vectorized_lookup(
        binned["_BIN_RANGE_"],
        var_woe_mapping["BIN_RANGE"],
        var_woe_mapping["WOE"],
    )
    missing_count = int(woe_values.isna().sum())
    if missing_count:
        logging.warning(f"WARNING: Failed to Map WOE values for {missing_count} Records!")

    return pd.DataFrame(
        {
            "_BIN_NUM_": binned["_BIN_NUM_"].to_numpy(copy=False),
            "_BIN_RANGE_": binned["_BIN_RANGE_"].to_numpy(copy=False),
            f"{var}{suffix}": woe_values.to_numpy(copy=False),
        },
        index=data.index,
    )


def _mapping_woe_single_var(
    data,
    var,
    woe_mapping_table,
    suffix="_woe",
    drop_bin_info=True,
    missing_ref_value=-999999,
):
    """Map WOE values for a single variable based on a WOE mapping table.

    Transform new data using a precomputed WOE mapping table,
    verifying monotonicity and checking the mapping result.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    var : str
        Name of the variable to map.
    woe_mapping_table : pd.DataFrame
        WOE mapping table.
    suffix : str, optional
        Suffix appended to the variable names. Default is '_woe'.
    drop_bin_info : bool, optional
        Whether to drop the intermediate bin information columns. Default is True.
    missing_ref_value : scalar, optional
        Training-time bin sentinel used when transforming missing values. Default is -999999.
        If a custom sentinel was used in training, the same value must be passed at transform time.

    Returns
    -------
    pd.DataFrame
        The DataFrame after mapping.

    Notes
    -----
    A warning is emitted when the mapping fails.
    """
#     from Modeling_Tool.Modeling_Tool.Model_Eval_Tool import run_binning
#     from code_optimization.Model_Eval_Tool import get_bin_range_list

    mapped = _map_woe_arrays(
        data,
        var,
        woe_mapping_table,
        suffix=suffix,
        missing_ref_value=missing_ref_value,
    )
    mapped_cols = [f"{var}{suffix}"]
    if not drop_bin_info:
        mapped_cols = ["_BIN_NUM_", "_BIN_RANGE_", *mapped_cols]

    base = data.drop(columns=mapped_cols, errors="ignore")
    return pd.concat([base, mapped[mapped_cols]], axis=1)


def mapping_woe(
    data,
    varlist,
    woe_mapping_table,
    suffix="_woe",
    drop_bin_info=True,
    missing_ref_value=-999999,
):
    """Map WOE values in batch based on a WOE mapping table.

    Apply the WOE transformation to multiple variables using a precomputed WOE mapping table.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    varlist : list
        List of variable names to map.
    woe_mapping_table : pd.DataFrame
        WOE mapping table.
    suffix : str, optional
        Suffix appended to the variable names. Default is '_woe'.
    drop_bin_info : bool, optional
        Whether to drop the intermediate bin information columns. Default is True.
    missing_ref_value : scalar, optional
        Training-time bin sentinel used when transforming missing values. Default is -999999.
        If a custom sentinel was used in training, the same value must be passed at transform time.

    Returns
    -------
    pd.DataFrame
        The DataFrame after mapping.

    Examples
    --------
    >>> result = mapping_woe(df, ['var1', 'var2'], woe_table)
    """
    woe_columns = {}
    last_bin_columns = None
    for var in varlist:
        mapped = _map_woe_arrays(
            data,
            var,
            woe_mapping_table,
            suffix=suffix,
            missing_ref_value=missing_ref_value,
        )
        woe_columns[f"{var}{suffix}"] = mapped[f"{var}{suffix}"].to_numpy(copy=False)
        last_bin_columns = mapped[["_BIN_NUM_", "_BIN_RANGE_"]]

    output_cols = list(woe_columns)
    if not drop_bin_info:
        output_cols.extend(["_BIN_NUM_", "_BIN_RANGE_"])
    base = data.drop(columns=output_cols, errors="ignore").copy()

    frames = [base]
    if not drop_bin_info and last_bin_columns is not None:
        frames.append(last_bin_columns)
    if woe_columns:
        frames.append(pd.DataFrame(woe_columns, index=data.index))
    return pd.concat(frames, axis=1)
