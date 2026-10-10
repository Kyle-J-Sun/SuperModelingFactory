import os
import pandas as pd
import numpy as np
import warnings

from .WOE_Plot_Tool import get_bivar_graph
from .WOE_Tool import _vectorized_group_slopes, mapping_woe, woe_transform

from Modeling_Tool.Core.Binning_Tool import run_binning, get_bin_range_list
from Modeling_Tool.Core.utils import _WOE_PURE_BIN_EPS, _calc_woe_iv_values
from Modeling_Tool._utils.sentinels import SMF_MISSING_BIN

def get_overall_woe_table(woe_master, data, varlist=None):
    """Build the WOE statistics table for the overall sample, with a structure aligned to the training-set mapping table.

    Each variable is re-binned on ``data`` with the bin edges of its fitted mapping table
    (``woe_master.woe_dict``), and the per-bin counts, WOE, IV and lift are recomputed on this sample.

    Parameters
    ----------
    woe_master : WOE_Master
        Fitted ``WOE_Master``. Its ``woe_dict``, ``varlist``, ``dep`` and ``missing_ref_value`` attributes are used.
    data : pandas.DataFrame
        Sample to profile. It must contain every variable in ``varlist`` and the target column ``woe_master.dep``.
    varlist : list of str or None, default None
        Variables to profile. ``None`` uses ``woe_master.varlist``. A variable without an entry in
        ``woe_master.woe_dict`` raises ``KeyError``.

    Returns
    -------
    pandas.DataFrame
        One row per bin, stacked over the variables of ``varlist``, with the columns ``VAR``, ``BIN_NUM``,
        ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``AVG_BAD``, ``WOE``, ``IV``, ``N_BAD``, ``N_GOOD``,
        ``BAD_PCT_PER_BIN``, ``GOOD_PCT_PER_BIN`` and ``LIFT``.

    Notes
    -----
    - Missing values are filled with ``woe_master.missing_ref_value`` before binning. When ``data`` has missing
      values, they are reported as a row whose ``BIN_NUM`` and ``BIN_RANGE`` are NaN (and whose ``MIN`` and ``MAX``
      equal ``missing_ref_value``).
    - ``BIN_NUM`` is renumbered by the new binning, so it is not guaranteed to match the numbers of the training
      mapping table, and the rows are not guaranteed to be in bin order. The interval closure (``(a, b]`` or
      ``[a, b)``) is copied from the first non-missing ``BIN_RANGE`` of the training table.
    - ``LIFT`` is ``AVG_BAD`` divided by the unweighted mean of ``AVG_BAD`` over the bins, not by the overall bad
      rate. A bin without bads (or without goods) gets a finite WOE: ``eps`` (1e-06) is added to its two shares, as in
      ``MonotoneWOEBinner``, and the WOE of every other bin is not smoothed.
    """
    if varlist is None:
        varlist = woe_master.varlist

    all_tables = []
    for var in varlist:
        var_map = woe_master.woe_dict[var]

        # Extract the bin edges and keep inf
        edges = get_bin_range_list(var_map, col="BIN_RANGE")
        custom_edges = sorted(set(list(edges) + [-np.inf, np.inf]))

        # Infer the interval closure from the training-set BIN_RANGE
        first_bin = var_map["BIN_RANGE"].dropna().iloc[0]
        include_lowest = first_bin[0] == '['
        right = first_bin[-1] == ']'

        fill_val = woe_master.missing_ref_value
        data_proc = data[[var, woe_master.dep]].copy()
        data_proc[var] = data_proc[var].fillna(fill_val)
        data_proc["_smf_good_n"] = data_proc[woe_master.dep].eq(0).astype(np.int64)

        # Bin with the training-set edges
        binned, _ = run_binning(
            data=data_proc,
            column=var,
            nbins=custom_edges,
            include_missing=True,
            equal_freq=False,
            bin_colnames=("_BIN_NUM", "_BIN_RANGE"),
            ascending=True,
            include_lowest=include_lowest,
            right=right,
            spec_values=[fill_val]
        )

        dep = woe_master.dep
        grp = binned.groupby(["_BIN_NUM", "_BIN_RANGE"], dropna=False)
        stats = grp.agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(var, "count"),
            AVG_BAD=(dep, "mean"),
            N_BAD=(dep, "sum"),
            N_GOOD=("_smf_good_n", "sum")
        ).reset_index()

        # Compute WOE / IV / LIFT
        stats["BAD_PCT_PER_BIN"] = stats["N_BAD"] / stats["N_BAD"].sum()
        stats["GOOD_PCT_PER_BIN"] = stats["N_GOOD"] / stats["N_GOOD"].sum()
        stats["WOE"], stats["IV"] = _calc_woe_iv_values(
            stats, "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", pure_bin_eps=_WOE_PURE_BIN_EPS
        )
        stats["LIFT"] = stats["AVG_BAD"] / stats["AVG_BAD"].mean()
        stats["VAR"] = var

        stats = stats.rename(columns={"_BIN_NUM": "BIN_NUM", "_BIN_RANGE": "BIN_RANGE"})

        cols = ["VAR", "BIN_NUM", "BIN_RANGE", "MIN", "MAX", "N",
                "AVG_BAD", "WOE", "IV", "N_BAD", "N_GOOD",
                "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", "LIFT"]
        stats = stats[cols]
        all_tables.append(stats)

    return pd.concat(all_tables, ignore_index=True)


def get_group_woe_table(woe_master, data, group, varlist=None):
    """Build the WOE summary, pivot, and detail tables for grouped samples.

    Each variable is re-binned on ``data`` with the bin edges of its fitted mapping table
    (``woe_master.woe_dict``), and the bin statistics are computed for every value of ``group``.

    Parameters
    ----------
    woe_master : WOE_Master
        Fitted ``WOE_Master``. Its ``woe_dict``, ``varlist``, ``dep`` and ``missing_ref_value`` attributes are used.
    data : pandas.DataFrame
        Sample to profile. It must contain ``group``, the target column ``woe_master.dep`` and every variable in
        ``varlist``.
    group : str
        Name of the grouping column (for example a month or a sample tag).
    varlist : list of str or None, default None
        Variables to profile. ``None`` uses ``woe_master.varlist``. A variable without an entry in
        ``woe_master.woe_dict`` raises ``KeyError``.

    Returns
    -------
    dict
        A dictionary with three DataFrames:

        - ``"summary"``: one row per variable and group value, with the columns ``<group>``, ``N``, ``IV``,
          ``KS_PER_BIN``, ``TOP_LIFT``, ``BTM_LIFT``, ``SLOPE``, ``direction`` and ``VAR``.
        - ``"pivot"``: the WOE of every bin and group, indexed by (``VAR``, ``BIN_NUM``, ``BIN_RANGE``) with one
          column per group value.
        - ``"detail"``: one row per variable, group and bin, with the columns ``<group>``, ``_BIN_NUM``,
          ``_BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``AVG_BAD``, ``N_BAD``, ``N_GOOD``, ``BAD_PCT_PER_BIN``,
          ``GOOD_PCT_PER_BIN``, ``WOE``, ``IV``, ``LIFT`` and ``VAR``.

    Notes
    -----
    - Missing values are filled with ``woe_master.missing_ref_value`` before binning.
    - The bad and good shares (``BAD_PCT_PER_BIN``, ``GOOD_PCT_PER_BIN``) use the totals of the whole sample (all
      groups) as denominators, not the totals of the group, so ``WOE`` and ``IV`` of a group cell are relative to the
      overall bad and good counts. ``LIFT`` divides ``AVG_BAD`` by its unweighted mean over all group-bin cells of
      the variable.
    - In ``summary``, ``IV`` is the sum of the cell IVs of the group, and ``KS_PER_BIN`` repeats ``TOP_LIFT`` (the
      largest ``LIFT`` of the group); it is not a KS statistic. ``SLOPE`` is the least-squares slope of ``AVG_BAD``
      over the position of the bin within the group (NaN when it is undefined), and ``direction`` is its sign
      (1, -1, or 0 when the slope is zero or NaN).
    """
    if varlist is None:
        varlist = woe_master.varlist

    summaries = []
    pivots_woe = []
    pivots_bad = []
    detail_list = []

    for var in varlist:
        var_map = woe_master.woe_dict[var]
        edges = get_bin_range_list(var_map, col="BIN_RANGE")
        custom_edges = sorted(set(list(edges) + [-np.inf, np.inf]))
        first_bin = var_map["BIN_RANGE"].dropna().iloc[0]
        include_lowest = first_bin[0] == '['
        right = first_bin[-1] == ']'
        fill_val = woe_master.missing_ref_value

        data_proc = data[[group, var, woe_master.dep]].copy()
        data_proc[var] = data_proc[var].fillna(fill_val)
        data_proc["_smf_good_n"] = data_proc[woe_master.dep].eq(0).astype(np.int64)

        binned, _ = run_binning(
            data=data_proc,
            column=var,
            nbins=custom_edges,
            include_missing=True,
            equal_freq=False,
            bin_colnames=("_BIN_NUM", "_BIN_RANGE"),
            ascending=True,
            include_lowest=include_lowest,
            right=right,
            spec_values=[fill_val]
        )

        dep = woe_master.dep
        # Aggregate by both bin and group
        grp = binned.groupby([group, "_BIN_NUM", "_BIN_RANGE"], dropna=False)
        stats = grp.agg(
            MIN=(var, "min"),
            MAX=(var, "max"),
            N=(var, "count"),
            AVG_BAD=(dep, "mean"),
            N_BAD=(dep, "sum"),
            N_GOOD=("_smf_good_n", "sum")
        ).reset_index()

        # Compute the WOE within each bin (global denominators)
        total_bad = stats["N_BAD"].sum()
        total_good = stats["N_GOOD"].sum()
        stats["BAD_PCT_PER_BIN"] = stats["N_BAD"] / total_bad
        stats["GOOD_PCT_PER_BIN"] = stats["N_GOOD"] / total_good
        stats["WOE"], stats["IV"] = _calc_woe_iv_values(
            stats, "BAD_PCT_PER_BIN", "GOOD_PCT_PER_BIN", pure_bin_eps=_WOE_PURE_BIN_EPS
        )
        stats["LIFT"] = stats["AVG_BAD"] / stats["AVG_BAD"].mean()
        stats["VAR"] = var

        # Build pivot tables: WOE and AVG_BAD values under each group
        pvt_woe = stats.pivot_table(index=["_BIN_NUM", "_BIN_RANGE"], columns=group, values="WOE")
        pvt_bad = stats.pivot_table(index=["_BIN_NUM", "_BIN_RANGE"], columns=group, values="AVG_BAD")

        # Summarize the overall metrics of each group
        sum_grp = stats.groupby(group).agg(
            N=("N", "sum"),
            IV=("IV", "sum"),
            KS_PER_BIN=("LIFT", "max"),  # simplified; can be refined further
            TOP_LIFT=("LIFT", "max"),
            BTM_LIFT=("LIFT", "min")
        ).reset_index()

        # Compute the monotonicity slope (optional)
        slopes = _vectorized_group_slopes(stats, group, "AVG_BAD")
        sum_grp["SLOPE"] = sum_grp[group].map(slopes)
        slope_values = sum_grp["SLOPE"].to_numpy()
        sum_grp["direction"] = np.select(
            [slope_values > 0, slope_values < 0], [1, -1], default=0
        )
        sum_grp["VAR"] = var

        summaries.append(sum_grp)
        pivots_woe.append(pvt_woe)
        pivots_bad.append(pvt_bad)
        detail_list.append(stats)

    final_pivot = pd.concat(pivots_woe, keys=varlist, names=["VAR", "BIN_NUM", "BIN_RANGE"])
    # To also return the AVG_BAD pivot table, merge it in or return it separately
    return {
        "summary": pd.concat(summaries, ignore_index=True),
        "pivot": final_pivot,
        "detail": pd.concat(detail_list, ignore_index=True)
    }


class WOE_Master(object):
    """WOE Master class for WOE fitting, transformation, adjustment and plotting.

    This class provides a complete WOE encoding workflow including:
    - Fitting WOE bins from data
    - Loading existing WOE mapping tables
    - Transforming new data with WOE
    - Updating and adjusting WOE bins
    - Plotting bivariate WOE comparison charts

    The constructor parameters are documented in ``__init__``.

    Attributes
    ----------
    train_data : pandas.DataFrame
        Training dataset.
    varlist : list of str
        Variable names for WOE transformation. ``load_mapping_table`` replaces it with the variables of the loaded
        table; ``update_woe`` does not change it.
    dep : str or None
        Target variable name.
    graph_save_dir : str
        Base directory for saving graphs.
    woe_suffix : str
        Suffix of the WOE column names created by ``transform``.
    missing_ref_value : int or float
        Reference value that replaces missing data when ``transform`` bins it.
    woe_dict : dict
        WOE mapping dictionary ``{variable: DataFrame}``, filled by ``fit``, ``update_woe`` and
        ``load_mapping_table``. It is empty until one of them has run.

    Notes
    -----
    Only numeric (and boolean) variables can be binned; ``fit`` raises ``TypeError`` for a string column.
    """

    def __init__(self, train_data, varlist, dep=None, graph_save_dir="", woe_suffix="_woe", missing_ref_value=SMF_MISSING_BIN, remove_exist_dir = False):
        """Initialize WOE_Master instance.

        Parameters
        ----------
        train_data : pandas.DataFrame
            Training dataset with the features and the target. It is stored without a copy. ``fit`` and
            ``update_woe`` bin its columns, and ``transform()`` encodes it when no data is passed.
        varlist : list of str
            Variables to perform the WOE transformation on.
        dep : str or None, default None
            Target variable name (binary, 1 = bad). Needed by ``fit``, ``update_woe`` and ``plot_bivar_graph``.
        graph_save_dir : str, default ""
            Base directory of the images written by ``plot_bivar_graph``.
        woe_suffix : str, default "_woe"
            Suffix appended to the variable name to name the WOE columns created by ``transform``.
        missing_ref_value : int or float, default SMF_MISSING_BIN
            Value that replaces missing data when ``transform`` bins it. The default ``SMF_MISSING_BIN`` is the
            smallest float64, which cannot collide with real data.
        remove_exist_dir : bool, default False
            If True, ``graph_save_dir`` is deleted recursively when the object is created (errors, for example a
            missing folder, are ignored silently).

        Raises
        ------
        ValueError
            If ``missing_ref_value`` equals a value found in the columns of ``varlist`` of ``train_data``.

        Warns
        -----
        UserWarning
            If ``missing_ref_value`` is a finite number with an absolute value below 1e10 (it may collide with real
            data).
        """
        self.train_data = train_data
        self.varlist = varlist
        self.dep = dep
        self.graph_save_dir = graph_save_dir
        self.woe_suffix = woe_suffix
        self.missing_ref_value = missing_ref_value
        self.woe_dict = {}
        self._validate_missing_ref_value()
        
        if remove_exist_dir:
            self.remove_folder(graph_save_dir)

    def _validate_missing_ref_value(self):
        try:
            val = float(self.missing_ref_value)
        except (TypeError, ValueError):
            val = np.nan
        if np.isfinite(val) and abs(val) < 1e10:
            warnings.warn(
                f"missing_ref_value={self.missing_ref_value} may collide with real data; "
                "consider Modeling_Tool.SMF_MISSING_BIN",
                UserWarning,
                stacklevel=2,
            )

        cols = [c for c in self.varlist if c in self.train_data.columns]
        if not cols:
            return
        collision = (self.train_data[cols] == self.missing_ref_value).any().any()
        if bool(collision):
            raise ValueError(
                f"missing_ref_value={self.missing_ref_value} collides with real values in "
                "training data; use SMF_MISSING_BIN or a truly unhittable sentinel"
            )
        
    @staticmethod
    def remove_folder(file_path):
        """Delete the specified folder.

        Recursively delete the folder at the given path together with all of its
        contents; do nothing (silently) if the folder does not exist.

        Parameters
        ----------
        file_path : str
            Path of the folder to delete.
            
        Examples
        --------
        >>> WOE_Master.remove_folder('/path/to/folder')
        """
        import shutil
        try:
            shutil.rmtree(file_path)
        except Exception:
            pass

    def load_mapping_table(self, mapping_table_csv):
        """Load WOE mapping table from CSV file or DataFrame.

        Parameters
        ----------
        mapping_table_csv : str or pandas.DataFrame
            Path to a CSV file, or a DataFrame object, that holds the mapping table. The table needs a ``VAR``
            column that names the variable of each row (the format written by ``save_mapping_table``).

        Returns
        -------
        None
            Updates the ``woe_dict`` and ``varlist`` attributes.

        Raises
        ------
        AttributeError
            If the input is neither a string path nor a DataFrame (a ``pathlib.Path`` is rejected too).

        Notes
        -----
        ``woe_dict`` is replaced, not merged: it gets one sub-table per distinct ``VAR`` value. ``varlist`` becomes the
        list of those variables in order of first appearance. ``train_data`` is not used, so a scoring object can be
        created with an empty ``train_data`` and an empty ``varlist``.
        """
        if isinstance(mapping_table_csv, str):
            woe_mapping_table = pd.read_csv(mapping_table_csv)
        elif isinstance(mapping_table_csv, pd.DataFrame):
            woe_mapping_table = mapping_table_csv
        else:
            raise AttributeError("Please give either CSV path or Pandas DataFrame as an input! ")

        varlist = woe_mapping_table['VAR'].unique().tolist()

        woe_dict = {}
        for var in varlist:
            single_var_woe = woe_mapping_table.query(f"VAR == '{var}'")
            woe_dict[var] = single_var_woe

        self.woe_dict = woe_dict
        self.varlist = varlist

    def fit(self, nbins=10, equal_freq=True, tree_binning_seed=None, chi2_config=None,
            precision=5, min_bin_prop=0.05, include_missing=True, fillna=None, spec_values=[],
            sv_min_bin_size=0.0, sv_small_policy="keep",
            sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
        """Fit WOE binning for variables in varlist.

        Parameters
        ----------
        nbins : int, default 10
            Number of bins (maximum). The count is capped at ``max(5, 1 / min_bin_prop)`` and never exceeds 20, so a
            larger value has no effect.
        equal_freq : bool, default True
            Use equal frequency binning (``False`` gives equal width bins). Decision-tree binning ignores it.
        tree_binning_seed : int or None, default None
            Random seed for tree-based binning. Any non-zero value switches to decision-tree bins; ``None`` and ``0``
            keep the quantile bins.
        chi2_config : tuple or None, default None
            Chi-square binning configuration ``(init_bins, p_value)``, for example ``(100, 0.95)``: the number of
            initial equal frequency bins and the confidence of the chi-square merge (a higher value gives fewer bins).
            ``None`` disables chi-square binning. A dict raises ``KeyError``.
        precision : int, default 5
            Numerical precision: number of decimals of the bin edges (values are rounded to it before binning).
        min_bin_prop : float, default 0.05
            Minimum bin proportion. It lowers the cap on the number of bins (see ``nbins``) and sets the minimum bin
            size of chi-square merging. It does not enforce a minimum size on quantile, equal width or tree bins.
        include_missing : bool, default True
            Include missing values in the binning. They are replaced with ``fillna`` (default ``missing_ref_value``) and
            get a bin of their own (``NaN`` in ``MIN`` and ``MAX``) on every binning path; the other bins are fitted on
            the real values. Up to 0.9.0 the equal-frequency bins counted the filled values in their quantiles, and with
            the default ``missing_ref_value`` (which the edge rounding overflowed to ``-inf``) the missing rows shared
            the lowest bin on every path. If False, rows with a missing value are dropped from the fit; ``transform``
            then sends missing values to the lowest bin.
        fillna : int, float or None, default None
            Value that stands in for missing data while the bins are fitted. ``None`` uses ``missing_ref_value``, which
            is also what ``transform`` fills with, so that a missing value is scored by the bin it was fitted in. Before
            this was applied to every binning path, a value other than -999999 only reached the chi-square step and
            ``transform`` then sent the missing rows to a different bin than the one they were fitted in.
        spec_values : list, default []
            Special values to handle. Each value becomes a bin edge, so it ends a bin of its own (together with any
            values between the previous edge and the special value).
        sv_min_bin_size : float, default 0.0
            Low-frequency special-value bin threshold as a share of all samples; 0.0 disables the fallback (legacy
            behaviour). It must be in [0.0, 1.0).
        sv_small_policy : str, default "keep"
            ``'keep'`` / ``'neutral'`` / ``'merge_missing'``. How a special-value or missing bin below
            ``sv_min_bin_size`` is treated: ``'keep'`` leaves it, ``'neutral'`` sets its WOE and IV to 0, and
            ``'merge_missing'`` merges its counts into the missing bin (falling back to ``'neutral'`` with a
            ``UserWarning`` when there is no missing bin).
        sv_woe_smoothing : str, default "none"
            ``'none'`` / ``'laplace'``. ``'laplace'`` shrinks the bad rate of the special-value and missing bins that
            are kept toward the overall bad rate before the WOE is computed (only when ``sv_smoothing_alpha`` > 0).
        sv_smoothing_alpha : float, default 0.0
            Laplace smoothing strength alpha (pseudo-counts). It must be >= 0; 0.0 leaves the WOE unchanged.

        Returns
        -------
        None
            Updates the ``woe_dict`` attribute.

        Raises
        ------
        TypeError
            If a variable of ``varlist`` is not numeric (for example a string column).
        ValueError
            If ``sv_small_policy``, ``sv_woe_smoothing``, ``sv_min_bin_size`` or ``sv_smoothing_alpha`` is invalid.

        Notes
        -----
        - ``woe_dict[var]`` is replaced for every variable of ``varlist``, and the tables of other variables are kept.
          Each table has the columns ``BIN_NUM``, ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``AVG_SCORE``, ``N_BAD``,
          ``N_GOOD``, ``AVG_BAD``, ``AVG_GOOD``, ``BAD_PCT_PER_BIN``, ``GOOD_PCT_PER_BIN``, ``LIFT``, ``WOE``, ``IV`` and
          ``VAR``. A bin without bads (or without goods) gets a finite WOE: ``eps`` (1e-06) is added to its two shares, as in
          ``MonotoneWOEBinner``; the WOE of every other bin is not smoothed.
        - Up to 0.9.0, ``chi2_config`` with the default ``include_missing=True`` and ``missing_ref_value`` raised
          ``ValueError: Bin edges must be unique``; it works now.
        """
        if fillna is None:
            fillna = self.missing_ref_value

        for var in self.varlist:
            woe_res = woe_transform(train_df=self.train_data,
                                    oot_df=None,
                                    var=var,
                                    dep=self.dep,
                                    nbins=nbins,
                                    chi2_config=chi2_config,
                                    tree_binning_seed=tree_binning_seed,
                                    precision=precision,
                                    min_bin_prop=min_bin_prop,
                                    include_missing=include_missing,
                                    equal_freq=equal_freq,
                                    ascending=True,
                                    fillna=fillna,
                                    spec_values=spec_values,
                                    sv_min_bin_size=sv_min_bin_size,
                                    sv_small_policy=sv_small_policy,
                                    sv_woe_smoothing=sv_woe_smoothing,
                                    sv_smoothing_alpha=sv_smoothing_alpha,
                                    drop_bin_info=True,
                                    ret_woe_table=True)
            woe_table = woe_res[-1]
            woe_table["VAR"] = var
            woe_table = woe_table.rename(columns={f"_bin_num_{var}": "BIN_NUM", f"_bin_range_{var}": "BIN_RANGE"})
            self.woe_dict[var] = woe_table

    def get_mapping_table(self):
        """Get WOE mapping table for all variables.

        Returns
        -------
        pandas.DataFrame
            WOE mapping information for all variables: the tables of ``woe_dict`` stacked in dictionary order, with the
            columns ``BIN_NUM``, ``BIN_RANGE``, ``MIN``, ``MAX``, ``N``, ``AVG_SCORE``, ``N_BAD``, ``N_GOOD``,
            ``AVG_BAD``, ``AVG_GOOD``, ``BAD_PCT_PER_BIN``, ``GOOD_PCT_PER_BIN``, ``LIFT``, ``WOE``, ``IV`` and ``VAR``.

        Raises
        ------
        ValueError
            If ``woe_dict`` is empty (``fit``, ``update_woe`` or ``load_mapping_table`` has not produced any table).
        """
        if not self.woe_dict:
            raise ValueError(
                "WOE_Master.woe_dict is empty — no variables were successfully binned. "
                "Check upstream: (1) varlist is non-empty, (2) train_data has non-null values, "
                "(3) any per-variable errors during .fit(). See .failed_variables if available."
            )
        return pd.concat([v for k, v in self.woe_dict.items()])

    def save_mapping_table(self, save_dir):
        """Save WOE mapping table as CSV file.

        Parameters
        ----------
        save_dir : str
            Path to save the CSV file. Despite its name it is the path of the file, not of a directory: it is passed to
            ``DataFrame.to_csv``, so its folder must already exist.

        Returns
        -------
        None
            Saves the file directly (the mapping table of ``get_mapping_table`` without the index).

        Raises
        ------
        ValueError
            If ``woe_dict`` is empty.
        """
        mapping_table = self.get_mapping_table()
        mapping_table.to_csv(save_dir, index=False)

    def transform(self, data=None, varlist=None):
        """Transform data using WOE encoding.

        Parameters
        ----------
        data : pandas.DataFrame or None, default None
            Data to transform (default: train_data). It must contain the raw columns of ``varlist``.
        varlist : list of str or None, default None
            Variables to transform (default: self.varlist).

        Returns
        -------
        pandas.DataFrame
            WOE transformed data: a copy of ``data`` with one ``<var><woe_suffix>`` column added per variable (an
            existing column of that name is replaced). The original columns are kept.

        Raises
        ------
        ValueError
            If ``woe_dict`` is empty.
        KeyError
            If a variable of ``varlist`` has no rows in the mapping table.

        Notes
        -----
        Missing values are replaced by ``missing_ref_value`` before they are binned, so they fall in the missing bin of
        the mapping table, or in the lowest bin when it has none (fitted with ``include_missing=False`` or on data
        without missing values). A value that falls in no bin of the mapping table gets
        a ``NaN`` WOE, and a warning ``Failed to Map WOE values for N Records`` is logged.
        """
        if data is None:
            data = self.train_data
        if varlist is None:
            varlist = self.varlist

        woe_mapping_table = self.get_mapping_table()
        data_woe = mapping_woe(
            data,
            varlist,
            woe_mapping_table,
            suffix=self.woe_suffix,
            drop_bin_info=True,
            missing_ref_value=self.missing_ref_value,
        )
        return data_woe

    def update_woe(self, varlist, nbins=10, equal_freq=True, tree_binning_seed=None, chi2_config=None,
                   precision=5, min_bin_prop=0.05, include_missing=True, fillna=None, spec_values=[],
                   sv_min_bin_size=0.0, sv_small_policy="keep",
                   sv_woe_smoothing="none", sv_smoothing_alpha=0.0):
        """Update WOE binning for specified variables.

        Parameters
        ----------
        varlist : list of str
            Variables to update WOE for. They are re-binned on ``train_data``; ``self.varlist`` is not changed, and the
            tables of the other variables stay as they are.
        nbins : int, default 10
            Number of bins (maximum). The count is capped at ``max(5, 1 / min_bin_prop)`` and never exceeds 20, so a
            larger value has no effect.
        equal_freq : bool, default True
            Use equal frequency binning (``False`` gives equal width bins). Decision-tree binning ignores it.
        tree_binning_seed : int or None, default None
            Random seed for tree-based binning. Any non-zero value switches to decision-tree bins; ``None`` and ``0``
            keep the quantile bins.
        chi2_config : tuple or None, default None
            Chi-square binning configuration ``(init_bins, p_value)``, for example ``(100, 0.95)``: the number of
            initial equal frequency bins and the confidence of the chi-square merge (a higher value gives fewer bins).
            ``None`` disables chi-square binning. A dict raises ``KeyError``.
        precision : int, default 5
            Numerical precision: number of decimals of the bin edges (values are rounded to it before binning).
        min_bin_prop : float, default 0.05
            Minimum bin proportion. It lowers the cap on the number of bins (see ``nbins``) and sets the minimum bin
            size of chi-square merging. It does not enforce a minimum size on quantile, equal width or tree bins.
        include_missing : bool, default True
            Include missing values in the binning. They are replaced with ``fillna`` (default ``missing_ref_value``) and
            get a bin of their own (``NaN`` in ``MIN`` and ``MAX``) on every binning path; the other bins are fitted on
            the real values. Up to 0.9.0 the equal-frequency bins counted the filled values in their quantiles, and with
            the default ``missing_ref_value`` (which the edge rounding overflowed to ``-inf``) the missing rows shared
            the lowest bin on every path. If False, rows with a missing value are dropped from the fit; ``transform``
            then sends missing values to the lowest bin.
        fillna : int, float or None, default None
            Value that stands in for missing data while the bins are fitted. ``None`` uses ``missing_ref_value``, which
            is also what ``transform`` fills with, so that a missing value is scored by the bin it was fitted in. Before
            this was applied to every binning path, a value other than -999999 only reached the chi-square step and
            ``transform`` then sent the missing rows to a different bin than the one they were fitted in.
        spec_values : list, default []
            Special values to handle. Each value becomes a bin edge, so it ends a bin of its own (together with any
            values between the previous edge and the special value).
        sv_min_bin_size : float, default 0.0
            Low-frequency special-value bin threshold as a share of all samples; 0.0 disables the fallback (legacy
            behaviour). It must be in [0.0, 1.0).
        sv_small_policy : str, default "keep"
            ``'keep'`` / ``'neutral'`` / ``'merge_missing'``. How a special-value or missing bin below
            ``sv_min_bin_size`` is treated: ``'keep'`` leaves it, ``'neutral'`` sets its WOE and IV to 0, and
            ``'merge_missing'`` merges its counts into the missing bin (falling back to ``'neutral'`` with a
            ``UserWarning`` when there is no missing bin).
        sv_woe_smoothing : str, default "none"
            ``'none'`` / ``'laplace'``. ``'laplace'`` shrinks the bad rate of the special-value and missing bins that
            are kept toward the overall bad rate before the WOE is computed (only when ``sv_smoothing_alpha`` > 0).
        sv_smoothing_alpha : float, default 0.0
            Laplace smoothing strength alpha (pseudo-counts). It must be >= 0; 0.0 leaves the WOE unchanged.

        Returns
        -------
        None
            Updates the ``woe_dict`` attribute.

        Raises
        ------
        TypeError
            If a variable of ``varlist`` is not numeric (for example a string column).
        ValueError
            If ``sv_small_policy``, ``sv_woe_smoothing``, ``sv_min_bin_size`` or ``sv_smoothing_alpha`` is invalid.

        Notes
        -----
        The new tables are merged into ``woe_dict`` only after every variable has been binned, so an error leaves
        ``woe_dict`` unchanged. A variable that is not yet in ``woe_dict`` is added to it (but not to ``self.varlist``).
        """
        if fillna is None:
            fillna = self.missing_ref_value

        new_woe_dict = {}
        for var in varlist:
            woe_res = woe_transform(train_df=self.train_data,
                                    oot_df=None,
                                    var=var,
                                    dep=self.dep,
                                    nbins=nbins,
                                    chi2_config=chi2_config,
                                    tree_binning_seed=tree_binning_seed,
                                    precision=precision,
                                    min_bin_prop=min_bin_prop,
                                    include_missing=include_missing,
                                    equal_freq=equal_freq,
                                    ascending=True,
                                    fillna=fillna,
                                    spec_values=spec_values,
                                    sv_min_bin_size=sv_min_bin_size,
                                    sv_small_policy=sv_small_policy,
                                    sv_woe_smoothing=sv_woe_smoothing,
                                    sv_smoothing_alpha=sv_smoothing_alpha,
                                    drop_bin_info=True,
                                    ret_woe_table=True)
            woe_table = woe_res[-1]
            woe_table["VAR"] = var
            woe_table = woe_table.rename(columns={f"_bin_num_{var}": "BIN_NUM", f"_bin_range_{var}": "BIN_RANGE"})
            new_woe_dict[var] = woe_table

        self.woe_dict.update(new_woe_dict)

    def plot_bivar_graph(self, data, group=None, dirname=None, varlist=None):
        """Plot bivariate WOE comparison graph.

        Parameters
        ----------
        data : pandas.DataFrame
            Data for plotting. It must contain the raw columns of ``varlist`` (not the WOE columns), the target column
            ``dep`` and, when ``group`` is given, the grouping column.
        group : str or None, default None
            Grouping variable name for distinguishing curves. When ``None`` a single ungrouped curve per variable is
            drawn and no ``by_<group>`` output (the ``<var>_<group>.png`` charts) is produced. Prior to 0.6.2 this was
            a required positional argument, which silently broke ungrouped callers (see ``_plot_woe`` in
            ``Modeling_Tool.Pipeline.feature_validation``). 0.6.3 also fixes ``WOE_Plot_Tool.get_bivar_graph``: the
            ``if group:`` guard there was moved up to wrap the summary/align/plot triplet. The guard used to sit
            *after* an unconditional ``data.groupby([None])`` that raised ``TypeError`` on ungrouped calls.
        dirname : str or None, default None
            Subdirectory name for saving, under ``graph_save_dir``. Required in practice: ``None`` raises
            ``TypeError``. It is kept keyword-only-in-practice by keeping the ``group`` slot second, for backward
            compatibility with any pre-0.6.2 positional caller.
        varlist : list of str or None, default None
            Variables to plot (default: self.varlist).

        Returns
        -------
        None
            Saves images directly.

        Raises
        ------
        TypeError
            If ``dirname`` is ``None``.
        ValueError
            If ``woe_dict`` is empty.

        Notes
        -----
        ``<graph_save_dir>/<dirname>/`` is created if it does not exist. The method writes ``<var>.png`` (the WOE of the
        mapping table) for every variable, plus ``<var>_<group>.png`` when ``group`` is given. In the grouped charts,
        rows with a missing value in the variable are dropped.
        """
        if dirname is None:
            raise TypeError("plot_bivar_graph() missing required argument: 'dirname'")
        if varlist is None:
            varlist = self.varlist

        save_dir = os.path.join(self.graph_save_dir, dirname)
        # matplotlib does not create parent directories on save. Historically
        # callers were expected to `make_dirs(save_dir)` first; feature_validation
        # only creates the base `figs/woe/<target>` and misses per-subdirectory
        # nesting. Since this method already owns the composed path, own the
        # `mkdir` too — avoids FileNotFoundError on the very first PNG write.
        os.makedirs(save_dir, exist_ok=True)

        get_bivar_graph(data=data,
                        varlist=varlist,
                        sep=self.dep,
                        ref_woe_table=self.get_mapping_table(),
                        group=group,
                        save_dir=save_dir)


# =============================================================================
# Standalone Function Wrappers
# =============================================================================

def load_mapping_table(mapping_table_csv):
    """Load WOE mapping table from CSV or DataFrame.

    Parameters
    ----------
    mapping_table_csv : str or pandas.DataFrame
        Path to a CSV file, or a DataFrame, that holds the mapping table. The table needs a ``VAR`` column that names
        the variable of each row.

    Returns
    -------
    tuple
        ``(varlist, woe_dict)``: the list of variables in order of first appearance in the ``VAR`` column, and the
        dictionary ``{variable: DataFrame}`` with the rows of each variable.

    Raises
    ------
    AttributeError
        If the input is neither a string path nor a DataFrame (a ``pathlib.Path`` is rejected too).
    """
    if isinstance(mapping_table_csv, str):
        woe_mapping_table = pd.read_csv(mapping_table_csv)
    elif isinstance(mapping_table_csv, pd.DataFrame):
        woe_mapping_table = mapping_table_csv
    else:
        raise AttributeError("Please give either CSV path or Pandas DataFrame as an input! ")

    varlist = woe_mapping_table['VAR'].unique().tolist()
    woe_dict = {}
    for var in varlist:
        single_var_woe = woe_mapping_table.query(f"VAR == '{var}'")
        woe_dict[var] = single_var_woe
    return varlist, woe_dict


def get_mapping_table(woe_dict):
    """Get combined mapping table from WOE dictionary.

    Parameters
    ----------
    woe_dict : dict
        WOE mapping dictionary ``{variable: DataFrame}``, for example the ``woe_dict`` attribute of a fitted
        ``WOE_Master``.

    Returns
    -------
    pandas.DataFrame
        WOE mapping information: the tables of ``woe_dict`` stacked in dictionary order.

    Raises
    ------
    ValueError
        If ``woe_dict`` is empty.
    """
    if not woe_dict:
        raise ValueError(
            "WOE_Master.woe_dict is empty — no variables were successfully binned. "
            "Check upstream: (1) varlist is non-empty, (2) train_data has non-null values, "
            "(3) any per-variable errors during .fit(). See .failed_variables if available."
        )
    return pd.concat([v for k, v in woe_dict.items()])


def save_mapping_table(woe_dict, save_dir):
    """Save WOE dictionary as CSV file.

    Parameters
    ----------
    woe_dict : dict
        WOE mapping dictionary ``{variable: DataFrame}``.
    save_dir : str
        Path to save the CSV file. Despite its name it is the path of the file, not of a directory: it is passed to
        ``DataFrame.to_csv``, so its folder must already exist.

    Returns
    -------
    None
        Saves the file directly (the stacked tables of ``woe_dict`` without the index).

    Raises
    ------
    ValueError
        If ``woe_dict`` is empty.
    """
    mapping_table = get_mapping_table(woe_dict)
    mapping_table.to_csv(save_dir, index=False)


def transform(data, varlist, woe_mapping_table, woe_suffix="_woe"):
    """Transform data using WOE encoding.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to transform. It must contain the raw columns of ``varlist``.
    varlist : list of str
        Variables to transform.
    woe_mapping_table : pandas.DataFrame
        WOE mapping table (for example ``WOE_Master.get_mapping_table()``), with the columns ``VAR``, ``BIN_NUM``,
        ``BIN_RANGE``, ``WOE`` and ``N``.
    woe_suffix : str, default "_woe"
        Suffix for WOE variable names: the WOE of ``var`` is stored in the column ``<var><woe_suffix>``.

    Returns
    -------
    pandas.DataFrame
        WOE transformed data: a copy of ``data`` with one ``<var><woe_suffix>`` column added per variable (an existing
        column of that name is replaced). The original columns are kept.

    Raises
    ------
    KeyError
        If a variable of ``varlist`` has no rows in ``woe_mapping_table``.

    Notes
    -----
    Missing values are replaced by -999999 (the default ``missing_ref_value`` of ``mapping_woe``) before they are
    binned. This function cannot pass another value: for a table made with a custom ``missing_ref_value``, call
    ``mapping_woe`` directly.
    """
    return mapping_woe(data, varlist, woe_mapping_table, suffix=woe_suffix, drop_bin_info=True)


def plot_bivar_graph_func(data, varlist, dep, ref_woe_table, group, save_dir):
    """Plot bivariate WOE comparison graph.

    Parameters
    ----------
    data : pandas.DataFrame
        Data for plotting. It must contain the raw columns of ``varlist``, the target column ``dep`` and, when
        ``group`` is given, the grouping column.
    varlist : list of str
        Variables to plot.
    dep : str
        Target variable name.
    ref_woe_table : pandas.DataFrame
        Reference WOE mapping table, with the rows of every variable of ``varlist`` (for example
        ``WOE_Master.get_mapping_table()``).
    group : str or None
        Grouping variable name for distinguishing curves. A falsy value (``None`` or an empty string) draws only the
        ungrouped chart of each variable. It has no default and must be passed.
    save_dir : str
        Directory path to save images. It must already exist (unlike ``WOE_Master.plot_bivar_graph``, this function
        does not create it).

    Returns
    -------
    None
        Saves images directly: ``<var>.png`` for every variable, plus ``<var>_<group>.png`` when ``group`` is given.
    """
    get_bivar_graph(data=data,
                    varlist=varlist,
                    sep=dep,
                    ref_woe_table=ref_woe_table,
                    group=group,
                    save_dir=save_dir)
