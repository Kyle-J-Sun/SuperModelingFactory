import logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, chi2
from Modeling_Tool._utils.frames import as_binning_numeric


def _round_edge(value, precision):
    """Round a bin edge to ``precision`` decimals, keeping an edge that the rounding would overflow to infinity.

    Python's ``round`` turns a finite value near the float limits (such as the ``-1.797e308`` missing sentinel of
    ``WOE_Master``) into ``-inf``; that edge then merges with the ``-inf`` edge and the bin it closes disappears.
    """
    rounded = round(value, precision)
    if not np.isfinite(rounded) and np.isfinite(value):
        return value
    return rounded


def _overflow_merges_edges(edges):
    """Whether rounding or pandas' interval labels would turn two distinct edges into the same infinity.

    A finite edge near the float limits overflows to ``-inf`` (``inf``) when it is rounded or formatted; that only loses a
    bin when the same infinity, or a second such edge of the same sign, is among the edges as well.
    """
    arr = np.asarray(edges, dtype=float)
    huge = np.isfinite(arr) & (np.abs(arr) > 1e300)
    if not bool(huge.any()):
        return False
    for sign in (-1.0, 1.0):
        n_huge = int((np.sign(arr[huge]) == sign).sum())
        has_inf = bool((np.isinf(arr) & (np.sign(arr) == sign)).any())
        if n_huge + int(has_inf) > 1:
            return True
    return False


logging.basicConfig(level=logging.INFO, format="%(message)s")

def get_max_nbins(data, nbins, min_bin_prop = 0.05):
    """
    Compute the maximum number of bins from a given minimum bin proportion.
    
    From the total number of records and the minimum bin proportion, compute the largest number of bins that still meets the minimum sample-size requirement.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table
    nbins : int
        Desired number of bins
    min_bin_prop : float, default 0.05
        Minimum proportion of samples in each bin
    
    Returns
    -------
    int
        Maximum feasible number of bins
    
    Examples
    --------
    >>> get_max_nbins(data, nbins=10, min_bin_prop=0.05)
    """
    
    n = data.shape[0]
    if n == 0:
        return nbins   # empty data cannot be binned; return the original value
    
    min_bin_size = min_bin_prop * n
    
    if min_bin_size == 0:
            # Here min_bin_prop == 0, which degenerates to at most nbins bins
            return min(nbins, n)
        
    nbins = min(nbins, max(5, n // min_bin_size))
    return nbins

def get_decision_tree_binning_edges(feature, target, max_leaf_nodes=5, min_samples_leaf=0.05, random_state=42, missing_ref_value = None, spec_values = []):
    """
    Find the optimal bins of a continuous variable using a decision tree.
    
    Use a decision-tree classifier to find the optimal bin edges; suitable for automated binning of continuous variables.
    The algorithm handles missing values and special values automatically and returns suitable bin edges.
    
    Parameters
    ----------
    feature : array-like
        Continuous feature to bin, a pandas Series or a 1-D array
    target : array-like
        Target variable, a binary label (pandas Series or 1-D array)
    max_leaf_nodes : int, default 5
        Maximum number of leaf nodes of the decision tree, i.e. the maximum number of bins
    min_samples_leaf : float, default 0.05
        Minimum sample fraction per leaf node; defaults to 0.05 (5%)
    random_state : int, default 42
        Random seed to make the results reproducible
    missing_ref_value : any, optional
        Reference value for missing values; this value is treated as missing
    spec_values : list, optional
        List of special values; these values are binned separately
    
    Returns
    -------
    bin_edges : list
        List of bin edges
    
    Examples
    --------
    >>> edges = get_decision_tree_binning_edges(feature, target, max_leaf_nodes=5)
    """
    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.utils import check_random_state
    
    if missing_ref_value:
        feature = np.where(feature == missing_ref_value, np.nan, feature)
    
    # Data preprocessing: remove missing values
    df = pd.DataFrame({
        'feature': feature,
        'target': target
    }).dropna()
    
    if len(spec_values) > 0:
        df = df[~df["feature"].isin(spec_values)]
    
    feature_clean = df['feature']
    target_clean = df['target']
    
    # With at most one distinct value there is nothing to split: return what the tree returns when it makes no
    # split (no thresholds, plus the missing reference), so that the caller sees a list of edges in every case.
    if feature_clean.nunique() <= 1:
        logger.info("Warning: feature variance is 0, cannot bin")
        return [missing_ref_value] if missing_ref_value else []
    
    # Reshape the feature to a 2D array to fit sklearn
    X = feature_clean.values.reshape(-1, 1)
    y = target_clean.values
    
    # Create the decision-tree classifier
    tree_model = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state
    )
    
    # Fit the decision-tree model
    tree_model.fit(X, y)
    
    # Extract the bin edges
    threshold = tree_model.tree_.threshold
    feature_min = feature_clean.min()
    feature_max = feature_clean.max()
    
    # Get the split thresholds of the non-leaf nodes and sort them
    bin_edges = sorted([th for th in threshold if th != -2])
    
    # Add the minimum and maximum values as edges
#     bin_edges = [feature_min] + bin_edges + [feature_max]
    
    # Remove possibly duplicated edges
    bin_edges = sorted(list(set(bin_edges)))
    if missing_ref_value:
        bin_edges = sorted(list(set(bin_edges + [missing_ref_value])))
        
#     print(tree_model.get_params())
    
    return bin_edges

class NumVarBinning:
    """
    Compute the cut-point sequence of a numeric variable from the number of bins and the binning method.
    
    The cut points are computed from the specified number of bins; the binning methods are "equal-width" and "equal-frequency":
    (1) When the number of bins is not smaller than the number of unique values of the variable, the cut-point sequence is the sequence of unique values
    (2) When the number of bins is smaller than the number of unique values, either "equal-width" or "equal-frequency" binning can be chosen
    
    If values that must be binned separately are specified:
    (1) Add the values to the cut-point sequence
    (2) Add the upper bound of each value = value + 0.1**(precision+1) to the cut-point sequence
    
    Cut-point precision:
    (1) If the variable takes integer values, the precision is 1 decimal place
    (2) If the variable takes floating-point values, the precision is the number of decimal places of the input
    
    For better generalization, the minimum of the cut-point sequence is replaced with -inf and +inf is added as the maximum
    
    Parameters
    ----------
    var_name : str
        Variable name
    spec_values : list, optional
        Special values to be binned separately
    spec_digit : int, default 3
        Number of decimal places kept for the upper bound of a special cut point; fixed to 1 if the variable is an integer, otherwise equal to spec_digit
    
    Examples
    --------
    >>> nvb = NumVarBinning(var_name='income', spec_values=[-9999])
    >>> binning = nvb.equi_binning(df, bins=10)
    """
    def __init__(self, var_name, spec_values=None, spec_digit=3):
        """
        Initialize the numeric-variable binning object.
        
        Parameters
        ----------
        var_name : str
            Variable name
        spec_values : list, optional
            Special values to be binned separately
        spec_digit : int, default 3
            Number of decimal places of precision for special values
        """
        self.var_name = var_name
        self.spec_values = spec_values
        self.spec_digit = spec_digit
        self.bins = None
        self.cut_points = None
        self.bin_names = None

    def calc_equi_cutpoints(self, df, bins=10, equi_method="equif"):
        """
        Compute the cut-point sequence of a numeric variable from the number of bins and the equi-binning method.
        
        1. Compute the cut points from the specified number of bins; the binning methods are "equal-width" and "equal-frequency"
        (1) When the number of bins is not smaller than the number of unique values of the variable, the cut-point sequence is the sequence of unique values
        (2) When the number of bins is smaller than the number of unique values, either "equal-width" or "equal-frequency" binning can be chosen
        
        2. If values that must be binned separately are specified:
        (1) Add the values to the cut-point sequence;
        (2) Add the upper bound of each value = value + 0.1**(precision+1) to the cut-point sequence;
        
        3. Cut-point precision:
        (1) If the variable takes integer values, the precision is 1 decimal place
        (2) If the variable takes floating-point values, the precision is the number of decimal places of the input
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        bins : int, default 10
            Number of bins
        equi_method : string, default "equif"
            Binning method, candidate values {"equid": equal-width, "equif": equal-frequency}
        
        Returns
        -------
        cut_points : numpy.array
            Cut-point sequence
        """
        var_series = df[self.var_name]

        if any(pd.isnull(var_series)):
            raise ValueError(f"{self.var_name}: the variable contains NaN values")

        if pd.api.types.is_integer_dtype(var_series):
            spec_digit = 1
        else:
            spec_digit = self.spec_digit

        # Compute the cut-point values from the specified number of bins
        cut_points = var_series.unique()
        if len(cut_points) <= bins:
            cut_points.sort()
        elif equi_method == "equif":
            pct = np.arange(1, bins) / bins
            cut_points = var_series.quantile(pct, interpolation="higher").values
        elif equi_method == "equid":
            min_value = min(var_series)
            max_value = max(var_series)
            step = (max_value - min_value) / bins
            cut_points = np.arange(start=min_value, stop=max_value, step=step)[1:]
        else:
            raise ValueError("Invalid equi_method value.")

        # Add the special values that need to be binned separately
        if bool(self.spec_values) and len(self.spec_values) > 0:
            self.spec_values.sort()
            spec_values_upper = [x + 0.1**spec_digit for x in self.spec_values]
            cut_points = np.append(cut_points, self.spec_values + spec_values_upper)
            cut_points.sort()
        
#         print("NumVarBinning Cut Points: ", cut_points)
        # Return info
        self.equi_method = equi_method

        return cut_points

    def modify_cutpoints(self, df, points):
        """
        Adjust the cut-point sequence.
        
        1. Cut-point precision:
        (1) If the variable takes integer values, the precision is 1 decimal place
        (2) If the variable takes floating-point values, the precision is the number of decimal places of the input
        
        2. Filter out cut points that fall outside the value range of the variable, and remove duplicates
        
        3. For better generalization, the minimum of the cut-point sequence is replaced with -inf and +inf is added as the maximum.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        points : array-like
            Cut-point sequence
        
        Returns
        -------
        cut_points : numpy.array
            Adjusted cut-point sequence
        """
        var_series = df[self.var_name]

        if pd.api.types.is_integer_dtype(var_series):
            spec_digit = 1
        else:
            spec_digit = self.spec_digit

        # Filter the cut points within the value range of the variable
        ## Added
        min_value_wo_spec_values = min(var_series[~var_series.isin(self.spec_values)])
        max_value_wo_spec_values = max(var_series[~var_series.isin(self.spec_values)])
        
        min_value = min(var_series)
        max_value = max(var_series)
#         print("(MIN, MAX): ", min_value, max_value)
        cut_points = list(filter(lambda x: min_value <= x <= max_value, points))
#         print("Modify Init Cut Points: ", cut_points)
        if bool(self.spec_values):
            spec_cut_points = list(filter(lambda x: min_value <= x <= max_value, self.spec_values))
#             print("Modify Spec Cut Points: ", spec_cut_points)
            
            ## Added
            # Near the float limits (the -1.797e308 missing sentinel) adding the step changes nothing: take the next
            # representable value so that the special value still ends a bin of its own
            spec_values_upper = [
                x + 0.1**spec_digit if x + 0.1**spec_digit != x else np.nextafter(x, np.inf) for x in self.spec_values
            ]
            
            cut_points.extend(spec_cut_points)
            
            ## Added
            cut_points.extend(spec_values_upper)

        cut_points = np.array(cut_points).astype("float")
        with np.errstate(over="ignore"):
            rounded_points = cut_points.round(spec_digit)
        # A point the rounding overflows to infinity (the -1.797e308 sentinel) is kept as it is
        cut_points = np.unique(np.where(np.isfinite(rounded_points) | ~np.isfinite(cut_points), rounded_points, cut_points))
        cut_points = np.insert(cut_points, 0, -np.inf)
        cut_points = np.append(cut_points, np.inf)
        
#         print("Final Cut Points Before Binning: ", cut_points)
        # Filter to the cut points that have samples
        binning_series = pd.cut(df[self.var_name], cut_points, right=False, labels=cut_points[:-1])
        binning_cnts = binning_series.value_counts()
        cut_points = np.array(binning_cnts.index[binning_cnts > 0])
        cut_points.sort()

        # If the first cut point is not in spec_values, set it to -inf
        if bool(self.spec_values):
            if cut_points[0] not in self.spec_values:
                cut_points[0] = -np.inf
        else:
            cut_points[0] = -np.inf
        
        # Add +inf as the maximum of the variable
        cut_points = np.append(cut_points, np.inf)

        return cut_points

    def apt_binning(self, df, points, modify=True):
        """
        Bin the variable using the specified cut-point sequence.
        
        Cut-point precision:
        (1) If the variable takes integer values, the precision is 1 decimal place
        (2) If the variable takes floating-point values, the precision is the number of decimal places of the input
        
        Filter out cut points that fall outside the value range of the variable
        
        For better generalization, the minimum of the cut-point sequence is replaced with -inf and +inf is added as the maximum.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        points : array-like
            Cut-point sequence
        modify : bool, default True
            Whether to adjust the cut-point sequence
        
        Returns
        -------
        binning_series : pandas.Categorical
            Binned series
        """
        if modify:
            self.cut_points = self.modify_cutpoints(df, points)
        else:
            self.cut_points = sorted(points)
        self.bins = len(self.cut_points) - 1

        binning_series = pd.cut(df[self.var_name], self.cut_points, right=False)
        self.bin_names = binning_series.values.categories
        # self.bin_names = {i: str(c) for i, c in zip(range(self.bins), binning_series.values.categories)}
        # binning_series = binning_series.values.rename_categories(range(self.bins))

        return binning_series

    def equi_binning(self, df, bins=10, equi_method="equif"):
        """
        Compute the cut-point sequence of a numeric variable from the number of bins and the equi-binning method, then bin the variable.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        bins : int, default 10
            Number of bins
        equi_method : string, default "equif"
            Binning method, candidate values {"equid": equal-width, "equif": equal-frequency}
        
        Returns
        -------
        binning_series : pandas.Categorical
            Binned series
        """
        logging.info(f"-------- [{self.var_name} : EquiX Binning] --------")
        logging.info(f"PARAMS: bins={bins}, equi_method={equi_method}")
        cut_points = self.calc_equi_cutpoints(df=df, bins=bins, equi_method=equi_method)
        binning_series = self.apt_binning(df=df, points=cut_points, modify=True)

        return binning_series

    def apply_binning(self, df):
        """
        Bin the variable using the saved cut-point sequence.
        
        Bin new data using the cut points previously computed by methods such as equi_binning or auto_binning.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        
        Returns
        -------
        binning_series : pandas.Categorical
            Binned series
        """
        if self.cut_points is None:
            raise ValueError("The class attribute self.cut_points is None.")
        binning_series = self.apt_binning(df, self.cut_points, modify=False)

        return binning_series
 
    def auto_binning(self, df, tgt_name, max_bins=10, min_prop_in_bin=0.05, equi_bins=200, equi_method="equif", init_points = None,  binning_criteria="chi2", chi2_p=0.95):
        """
        Automatic binning based on the chi-square test.
        
        Using a distribution-distance measure, automatically derive the bin sequence for the given maximum number of bins and minimum number of samples per bin.
        Steps:
        1. Use the initial binning method to complete the initial binning
        2. Based on the distribution-distance measure, compute the sample size and within-bin distribution distance of each initial bin, as well as the gain in distance difference relative to the original bins after merging adjacent bins
        3. Keep merging bins whose sample size does not meet the minimum-samples-per-bin requirement and bins whose merge with an adjacent bin yields a larger distance-difference gain, until the maximum number of bins and the minimum number of samples per bin are satisfied
        4. Generate the binned variable series from the final binning result.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Data table
        tgt_name : str
            Target variable name
        max_bins : int, default 10
            Maximum number of bins
        min_prop_in_bin : float, default 0.05
            Minimum proportion of samples per bin
        equi_bins : int, default 200
            Number of initial bins
        equi_method : string, default "equif"
            Initial binning method, candidate values {"equid": equal-width, "equif": equal-frequency}
        init_points : array-like, default None
            Initial cut-point sequence; if not None, equi_bins and equi_method are ignored
        binning_criteria : string, default "chi2"
            Binning criterion, candidate values {"chi2": "chi-square value"}
        chi2_p : float, default 0.95
            Probability used to compute the quantile of the chi-square distribution with 1 degree of freedom. When the chi-square value of adjacent bins is smaller than this quantile, the bins are considered not independent and are merged; a larger value therefore makes the independence test stricter
        
        Returns
        -------
        binning_series : pandas.Categorical
            Binned series
        """
        logging.info(f"\n---------------- [{self.var_name} : Auto binning] ----------------")
        min_cnt_in_bin = np.floor(df.shape[0] * min_prop_in_bin)
        logging.info(f"PARAMS: max_bins={max_bins}, min_cnt_in_bin={min_cnt_in_bin}, criteria={binning_criteria}")

        # Initial binning
        if init_points is None:
            binning_series = self.equi_binning(df, bins=equi_bins, equi_method=equi_method)
        else:
            binning_series = self.apt_binning(df=df, points=init_points, modify=True)
        cut_points = self.cut_points

        # Pivot table of the initial bins
        df_tmp = df[[self.var_name, tgt_name]].copy()
        df_tmp[self.var_name] = binning_series
        df_pvt = cre_pvt(df=df_tmp, var_name=self.var_name, tgt_name=tgt_name)
        df_pvt.index = cut_points[:-1]

        # List the special values separately
        if bool(self.spec_values):
            df_pvt_spec = df_pvt.loc[[x in self.spec_values for x in df_pvt.index], ].copy()
            df_pvt = df_pvt.loc[[x not in self.spec_values for x in df_pvt.index], ].copy()
            max_bins = max_bins - df_pvt_spec.shape[0]
        
        # Automatic binning
        if not df_pvt.empty:
            logging.info(f"-------- [{self.var_name} : Auto binning] --------")
            df_pvt = chi2_auto_binning(df_pvt=df_pvt, max_bins=max_bins, min_cnt_in_bin=min_cnt_in_bin, p=chi2_p)
                        
            # Recompute the bins
            cut_points = df_pvt.index
            
            # Add the special values that need to be binned separately
            if bool(self.spec_values) and len(self.spec_values) > 0:
#                 spec_values_upper = [x + 0.1**spec_digit for x in self.spec_values]
                cut_points = np.append(cut_points, self.spec_values)
#                 cut_points = np.append(cut_points, self.spec_values_upper)
#                 cut_points.sort()
                
#             print("Cut Points After Auto Binning: ", cut_points)
        
        binning_series = self.apt_binning(
            df=df, 
            points=cut_points, 
            modify=True
            )

        return binning_series


def cre_pvt(df, var_name, tgt_name):
    """
    Create a pivot table with the variable as rows and the target variable as columns.
    
    Build a per-bin statistics pivot table from the specified variable and target variable, containing the number of negative samples,
    the number of positive samples, the total number of samples, and the target rate of each bin.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data table
    var_name : str
        Variable name (the grouping key)
    tgt_name : str
        Target variable name (binary label, 0 and 1)
    
    Returns
    -------
    df_pvt : pandas.DataFrame
        Variable pivot table, containing the columns:
        - 0: number of negative samples
        - 1: number of positive samples
        - n: total number of samples
        - tr: target rate (proportion of positive samples)
    
    Examples
    --------
    >>> df_pvt = cre_pvt(df, var_name='income', tgt_name='default')
    """
    
    df_pvt = df.groupby(var_name, observed=False)[tgt_name].value_counts().unstack().fillna(0)
    df_pvt["n"] = df_pvt[0] + df_pvt[1]
    df_pvt["tr"] = df_pvt[1] / df_pvt["n"]

    return df_pvt


def merge_bins(df_pvt, ilocs):
    """
    Merge variable bins based on position indices.
    
    Merge the bins at the specified positions, compute the statistics after merging (sample counts, target rate, etc.),
    and update the chi-square values and target-rate differences.
    
    Important: the index of df_pvt must not be reset
    
    Parameters
    ----------
    df_pvt : pandas.DataFrame
        Variable pivot table generated by the cre_pvt function
    ilocs : list
        List of position indices to merge (e.g. [0,1] merges the first two bins)
    
    Returns
    -------
    df_pvt_new : pandas.DataFrame
        Variable pivot table after merging
    
    Examples
    --------
    >>> df_pvt = merge_bins(df_pvt, ilocs=[0, 1])
    """
    ilocs.sort()
    df = df_pvt.copy()

    # Aggregate the field values into the smallest position index
    idxes = df.index
    l_idx = idxes[ilocs[0]]
    df.loc[l_idx, [0, 1, "n"]] = np.apply_over_axes(np.sum, df.loc[idxes[ilocs], [0, 1, "n"]], axes=0)[0]
    df.loc[l_idx, "tr"] = df.loc[l_idx, 1] / df.loc[l_idx, "n"]
    df.drop(index=idxes[ilocs[1:]], inplace=True)

    # Recompute tr_diff and chisq
    idxes = df.index
    if "tr_diff" in df.columns:
        if ilocs[0]+1 < df.shape[0]:
            df.loc[idxes[ilocs[0]], "tr_diff"] = df.loc[idxes[ilocs[0]+1], "tr"] - df.loc[idxes[ilocs[0]], "tr"]
        else:
            df.loc[idxes[ilocs[0]], "tr_diff"] = 0

    if "chisq" in df.columns:
        if ilocs[0]+1 < df.shape[0]:
            df.loc[idxes[ilocs[0]], "chisq"] = chi2_contingency(observed_laplace(df.loc[idxes[[ilocs[0], ilocs[0]+1]], [0, 1]]), correction=False)[0]
        else:
            df.loc[idxes[ilocs[0]], "chisq"] = np.inf

    return df


def observed_laplace(observed, digit=6):
    """
    Apply a Laplace correction to the values of a contingency table.
    
    Add a small amount (0.1**digit) to every cell of the contingency table to avoid
    chi-square test problems caused by zero counts.
    
    Parameters
    ----------
    observed : array_like
        Contingency table (a 2-D array or similar structure)
    digit : int, default 6
        Number of decimal places of the small amount added to the observed values; if -1, the integer 1 is added
    
    Returns
    -------
    obs_laplace : numpy.ndarray
        Contingency table after the Laplace correction
    
    Examples
    --------
    >>> observed = [[10, 20], [30, 40]]
    >>> obs_laplace = observed_laplace(observed, digit=6)
    """
    obs_laplace = np.asarray(observed) + 0.1**digit

    return obs_laplace


def cat_2_list(bin_series):
    """
    Convert a binned Categorical series to a list of edge values.
    
    Extract all interval edges from Categorical data and return the de-duplicated list
    made up of all left and right edges.
    
    Parameters
    ----------
    bin_series : pandas.Categorical
        Binned Categorical series
    
    Returns
    -------
    list
        List containing all interval edge values
    
    Examples
    --------
    >>> edges = cat_2_list(binning_series)
    """
    
    interval_list = bin_series.cat.categories.tolist()
    edges_res = set()
    for x in interval_list:
        edges_res.add(x.left)
        edges_res.add(x.right)
    edges_res = list(edges_res)
    
    return edges_res


def get_bin_range(edges, precision = 5, ascending = False, left_sign = '(', right_sign = ']'):
    """
    Generate a list of interval string descriptions from the bin edges.
    
    Convert a list of bin edges into a list of string descriptions with interval signs; supports custom
    open/closed interval signs and precision.
    
    Parameters
    ----------
    edges : array-like
        List of bin edge values
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    ascending : bool, default False
        Whether to sort in ascending order
    left_sign : str, default '('
        Left interval sign; '[' means inclusive, '(' means exclusive
    right_sign : str, default ']'
        Right interval sign; ']' means inclusive, ')' means exclusive
    
    Returns
    -------
    list
        List of interval string descriptions
    
    Examples
    --------
    >>> edges = [0, 10, 20, 30]
    >>> ranges = get_bin_range(edges, precision=0, ascending=True)
    ['[0, 10]', '[10, 20]', '[20, 30]']
    """
    
    i = 0
    reverse = not ascending
    # Very large edges (such as the float maximum) overflow to inf when rounded to the given precision, as they always did;
    # an edge whose overflow would merge it with another one (the -1.797e308 missing sentinel next to -inf) is kept
    round_edge = _round_edge if _overflow_merges_edges(edges) else round
    with np.errstate(over="ignore"):
        edges = sorted([round_edge(x, precision) for x in edges], reverse = reverse)
    res = []
    while i < len(edges) - 1:
        left = edges[i]
        right = edges[i+1]
        res.append(f"{left_sign}{left}, {right}{right_sign}")
        left = right
        i += 1
        
    return res


def _materialize_bin_columns(data, binned, bin_range_list, bin_num_col, bin_range_col):
    """Attach categorical bin numbers and labels without Python row callbacks."""
    # The two bin columns are ours to add: work on our own frame so callers that
    # pass a slice or their own DataFrame never get these columns written back.
    data = data.copy(deep = False)
    codes = binned.cat.codes.to_numpy(dtype=np.intp, copy=False)
    range_values = np.empty(len(codes), dtype=object)
    range_values[:] = np.nan
    valid = codes >= 0
    if valid.any():
        labels = np.asarray(bin_range_list, dtype=object)
        if codes[valid].max() >= len(labels):
            raise ValueError("Binning category codes do not match bin range labels")
        range_values[valid] = np.take(labels, codes[valid])

    data[bin_num_col] = binned.astype(object)
    data[bin_range_col] = range_values
    return data


def _parse_bin_range_bounds(data, col="_bin_range"):
    """Parse interval labels into numeric left and right bounds vectorially."""
    if col not in data.columns:
        raise KeyError(f"Column {col!r} is not present in the bin mapping table")

    ranges = data[col].astype("string").str.strip()
    bounds = ranges.str.extract(
        r"^[\[\(]\s*([^,]+?)\s*,\s*([^\]\)]+?)\s*[\]\)]$",
        expand=True,
    )
    invalid = bounds.isna().any(axis=1)
    if invalid.any():
        examples = ranges.loc[invalid].dropna().head(3).tolist()
        raise ValueError(f"Unable to parse bin range labels in {col!r}: {examples}")

    try:
        left = bounds.iloc[:, 0].str.strip().astype(float).to_numpy()
        right = bounds.iloc[:, 1].str.strip().astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Bin range column {col!r} contains non-numeric bounds") from exc
    return left, right


def get_bin_range_list(data, col = "_bin_range"):
    """
    Convert a column of bin interval strings to a list of edge values.
    
    Parse the bin interval string column of a DataFrame (such as "[0, 10)", "(10, 20]"),
    extract all unique edge values, and return them sorted.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the bin interval column
    col : str, default "_bin_range"
        Name of the bin interval column
    
    Returns
    -------
    list
        De-duplicated, sorted list of edge values
    
    Examples
    --------
    >>> unique_range = get_bin_range_list(data, col="bin_range")
    """
    
    left, right = _parse_bin_range_bounds(data, col=col)
    return np.unique(np.concatenate([left, right])).tolist()


def chi2_auto_binning(df_pvt, max_bins, min_cnt_in_bin, p=0.95):
    """
    Automatic binning based on the chi-square test.
    
    Use the chi-square test to decide whether adjacent bins should be merged, iterating through the following steps:
    1. First handle the head and tail bins whose sample size does not meet the minimum requirement
    2. Then handle the middle bins with insufficient sample size
    3. Finally, merge adjacent bins that are not independent according to the chi-square test
    
    Parameters
    ----------
    df_pvt : pandas.DataFrame
        Variable pivot table generated by the cre_pvt function, containing the columns: 0, 1, n, tr
    max_bins : int
        Maximum number of bins
    min_cnt_in_bin : int
        Minimum number of samples per bin
    p : float, default 0.95
        Probability used to compute the quantile of the chi-square distribution with 1 degree of freedom. When the chi-square value of adjacent bins is smaller than this quantile,
        the bins are considered not independent and are merged. The larger the value, the stricter the independence test
    
    Returns
    -------
    df_pvt : pandas.DataFrame
        Variable pivot table after merging
    
    Examples
    --------
    >>> df_pvt = chi2_auto_binning(df_pvt, max_bins=10, min_cnt_in_bin=100, p=0.95)
    """
    # Check whether the sample count meets the minimum-samples requirement
    if np.sum(df_pvt["n"]) <= min_cnt_in_bin:
        df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(df_pvt.shape[0])))
        logging.info("Merge ilocs: all")
    else:
        # Head and tail bins
        # Check whether the bins at the head and tail ends meet the minimum sample requirement
        csumn_asc = np.cumsum(df_pvt["n"])
        ilocs_head = np.min(np.where(csumn_asc >= min_cnt_in_bin))
        if ilocs_head > 0:
            ori_bins = df_pvt.shape[0]
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(ilocs_head+1)))
            logging.info(f"HeadMerge: ilocs={[0, ilocs_head+1]}, bins={ori_bins} -> {df_pvt.shape[0]}")
        
        csumn_desc = np.cumsum(df_pvt["n"][::-1])
        ilocs_tail = np.min(np.where(csumn_desc >= min_cnt_in_bin))
        if ilocs_tail > 0:
            ori_bins = df_pvt.shape[0]
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=list(range(ori_bins-(ilocs_tail+1), ori_bins)))
            logging.info(f"TailMerge: ilocs={[ori_bins-(ilocs_tail+1), ori_bins]}, bins={ori_bins} -> {df_pvt.shape[0]}")


        # Automatic binning
        # Compute, for each initial bin, the chi-square value, the distribution divergence after merging with the next bin, and the gain
        chisq = [chi2_contingency(observed_laplace(df_pvt.loc[df_pvt.index[[i, i+1]], [0, 1]]), correction=False)[0] for i in range(df_pvt.shape[0] - 1)]
        chisq.append(np.inf)
        tr_diff = np.diff(df_pvt["tr"])
        tr_diff = np.append(tr_diff, 0)
        df_pvt["chisq"] = chisq
        df_pvt["tr_diff"] = tr_diff

        r = 0
        ori_bins = df_pvt.shape[0]
        while df_pvt.shape[0] > max_bins or any(df_pvt["n"] < min_cnt_in_bin):
            r += 1
            # Handle first the bins whose sample size does not meet the minimum, merging forward or backward
            if any(df_pvt["n"] < min_cnt_in_bin):
                iloc = np.min(np.where(df_pvt["n"] < min_cnt_in_bin))
                # Determine the merge direction of the bin: position and monotonicity
                if iloc == 0:
                    iloc_merge = iloc + 1
                elif iloc == df_pvt.shape[0]-1:
                    iloc_merge = iloc - 1
                else:
                    idxes = df_pvt.index
                    if np.abs(df_pvt.loc[idxes[iloc-1], "tr_diff"]) <= np.abs(df_pvt.loc[idxes[iloc], "tr_diff"]):
                        iloc_merge = iloc - 1
                    else:
                        iloc_merge = iloc + 1
            # Then handle the bins with low chi-square values, merging backward
            else:
                chisq_list = df_pvt["chisq"].to_list()
                iloc = chisq_list.index(np.min(chisq_list))
                iloc_merge = iloc + 1
            
            # Merge the variable bins
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=[iloc, iloc_merge])
        logging.info(f"LoopMerge: round={r}, bins={ori_bins} -> {df_pvt.shape[0]}")


        # Chi-square test binning
        r = 0
        ori_bins = df_pvt.shape[0]
        while df_pvt["chisq"].min() <= chi2.ppf(p, 1):
            r += 1
            chisq_list = df_pvt["chisq"].to_list()
            iloc = chisq_list.index(np.min(chisq_list))
            df_pvt = merge_bins(df_pvt=df_pvt, ilocs=[iloc, iloc + 1])
        logging.info(f"Chi2Merge: round={r}, bins={ori_bins} -> {df_pvt.shape[0]}")

    return df_pvt


def quick_binning(data, column, labels = None, nbins = 10, precision = 5, equal_freq = True, right = True, include_lowest = False, 
                  min_bin_prop = 0.05, ascending = True, include_missing = False, tree_binning = False, target = None, random_state=42, 
                  fillna = -999999, spec_values = []):
    """
    Quick binning function.
    
    Quickly bin the data with equal-frequency or equal-width binning; several binning strategies are supported:
    - Equal-frequency binning: split by quantiles so that every bin has a similar number of samples
    - Equal-width binning: split evenly by value intervals
    - Decision-tree binning: use a decision tree to find the optimal cut points
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table
    column : str
        Name of the column to bin
    labels : array-like, optional
        Custom bin labels
    nbins : int or list/tuple, default 10
        Number of bins (an integer) or the specified bin edges (a list or tuple)
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning
    right : bool, default True
        Whether the intervals are closed on the right
    include_lowest : bool, default False
        Whether to include the lowest value
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin
    ascending : bool, default True
        Whether the bin order is ascending
    include_missing : bool, default False
        Whether to include missing values
    tree_binning : bool, default False
        Whether to use decision-tree binning
    target : str, optional
        Target variable name (required for decision-tree binning)
    random_state : int, default 42
        Random seed
    fillna : any, default -999999
        Fill value for missing values
    spec_values : list, default []
        List of special values; each is placed in its own bin
    
    Returns
    -------
    binned : pandas.Categorical
        Binned series
    bin_edges : numpy.ndarray
        Array of bin edges
    
    Examples
    --------
    >>> binned, edges = quick_binning(data, 'income', nbins=10, equal_freq=True)
    """
    
    # values near the float limits overflow when rounded, as they always did
    with np.errstate(over="ignore"):
        binning_series = as_binning_numeric(data[column]).round(precision)
    
    if include_missing:
        binning_series = binning_series.fillna(fillna)
    else:
        binning_series = binning_series.dropna()
        
    # Determine Binning Intervals
    if isinstance(nbins, int):
        
        value_no_spec_value = binning_series[~binning_series.isin(spec_values)]
#         print("Tree Binning No Spec Value: ", value_no_spec_value)

        if len(value_no_spec_value) == 0:
            # Return a default empty binning result
            return pd.Series(), []
        
        if equal_freq:
            nbins = int(get_max_nbins(data, nbins, min_bin_prop))
            # The missing rows (filled with `fillna`) get a bin of their own, as in the equal-width branch: they must not
            # take part in the quantiles, or they join the lowest bin together with real values
            real_values = value_no_spec_value
            missing_edge = []
            if include_missing:
                is_missing = value_no_spec_value == fillna
                if bool(is_missing.any()):
                    real_values = value_no_spec_value[~is_missing]
                    missing_edge = [fillna]
            breakpoints = (
                np.percentile(real_values, [100 / nbins * i for i in range(1, nbins)]) if len(real_values) else []
            )
            breakpoints = list(breakpoints) + missing_edge + spec_values
        else:
            nbins = int(get_max_nbins(data, nbins, min_bin_prop))
            min_value = binning_series.replace(fillna, np.nan).min() if include_missing else value_no_spec_value.min()
#             print("MIN Value: ", min_value)
            breakpoints = np.linspace(min_value, value_no_spec_value.max(), nbins + 1)[1:-1]
            breakpoints = np.sort(np.unique(list(breakpoints) + [fillna])) if include_missing else breakpoints
            breakpoints = list(breakpoints) + spec_values
            
        if tree_binning:
            breakpoints = get_decision_tree_binning_edges(
                                binning_series, 
                                data[target],
                                max_leaf_nodes = nbins,
                                min_samples_leaf = min_bin_prop,
                                random_state = random_state,
                                missing_ref_value = fillna if include_missing else None,
                                spec_values = spec_values
                            )
            breakpoints = list(breakpoints)
#         print("Tree Output: ", breakpoints)
        breakpoints = [_round_edge(x, precision) for x in breakpoints]
        fnl_breakpoints = np.sort(np.unique([-np.inf, *breakpoints, np.inf]))
#         print("Final Tree Output: ", fnl_breakpoints)
        
    if isinstance(nbins, list) or isinstance(nbins, tuple):
        fnl_breakpoints = nbins
    
    if len(spec_values) > 0:
        fnl_breakpoints = sorted(list(set(list(fnl_breakpoints) + spec_values)))
        
    # pandas rounds the edges to format interval labels; float-max edges overflow
    # to inf there as they always did, so the numpy notice is not useful
    if labels is None and _overflow_merges_edges(fnl_breakpoints):
        # Edges near the float limits (the -1.797e308 missing sentinel) would collapse into the same interval label and
        # merge their bins: number the bins instead (the callers label them from the exact edges)
        labels = list(range(len(fnl_breakpoints) - 1))
    with np.errstate(over="ignore"):
        binned, bin_edges = pd.cut(
            binning_series, 
            bins = fnl_breakpoints, 
            labels = labels, 
            right = right, 
            include_lowest = include_lowest, 
            retbins = True
        )
    
    orig_cat = [x for x in binned.cat.categories.tolist()]
    if ascending:
        binned = binned
    else:
        orig_cat.reverse()
        binned = binned.cat.reorder_categories([*orig_cat], ordered=True)
                                
    return binned, bin_edges


class Binning:
    """
    Unified binning class.
    
    Integrate several binning methods, such as quick binning and chi-square binning, behind a unified interface for binning data.
    Supports equal-frequency/equal-width binning, decision-tree binning, automatic chi-square binning, and other strategies.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table (it is copied; the original data is not modified)
    column : str
        Name of the column to bin
    tgt_name : str, optional
        Target variable name (required for chi-square binning)
    nbins : int, default 10
        Number of bins
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin
    include_missing : bool, default True
        Whether to include missing values
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        Tuple of the column names of the binning results
    ascending : bool, default True
        Whether the bin order is ascending
    right : bool, default True
        Whether the intervals are closed on the right
    include_lowest : bool, default False
        Whether to include the lowest value
    tree_binning : bool, default False
        Whether to use decision-tree binning
    chi2_method : bool, default False
        Whether to use chi-square binning
    chi2_p : float, default 0.95
        Significance level of the chi-square test
    init_equi_bins : int, default 200
        Number of initial equal-frequency bins
    fillna : any, default -999999
        Fill value for missing values
    spec_values : list, default []
        List of special values
    random_state : int, default 42
        Random seed
    
    Attributes
    ----------
    result : pandas.DataFrame
        Binning result data
    bin_edges : numpy.ndarray
        Array of bin edges
    
    Examples
    --------
    >>> binner = Binning(data, column='income', tgt_name='default', nbins=10)
    >>> binner.run()
    >>> result, edges = binner.get_result()
    
    >>> # Use chi-square binning
    >>> binner = Binning(data, column='income', tgt_name='default', 
    ...                  nbins=10, chi2_method=True, chi2_p=0.95)
    >>> binner.run()
    """
    
    def __init__(self, data, column, tgt_name=None, nbins=10, precision=5, min_bin_prop=0.05,
                 include_missing=True, equal_freq=True, bin_colnames=("_bin_num", "_bin_range"),
                 ascending=True, right=True, include_lowest=False, tree_binning=False,
                 chi2_method=False, chi2_p=0.95, init_equi_bins=200, fillna=-999999,
                 spec_values=[], random_state=42):
        """
        Initialize the binning object.
        
        Parameters
        ----------
        data : pandas.DataFrame
            Input data table
        column : str
            Name of the column to bin
        tgt_name : str, optional
            Target variable name
        nbins : int, default 10
            Number of bins
        precision : int, default 5
            Precision of the edge values
        min_bin_prop : float, default 0.05
            Minimum proportion of samples per bin
        include_missing : bool, default True
            Whether to include missing values
        equal_freq : bool, default True
            Whether to use equal-frequency (True) or equal-width (False) binning
        bin_colnames : tuple, default ("_bin_num", "_bin_range")
            Column names of the binning results
        ascending : bool, default True
            Whether the bin order is ascending
        right : bool, default True
            Whether the intervals are closed on the right
        include_lowest : bool, default False
            Whether to include the lowest value
        tree_binning : bool, default False
            Whether to use decision-tree binning
        chi2_method : bool, default False
            Whether to use chi-square binning
        chi2_p : float, default 0.95
            Significance level of the chi-square test
        init_equi_bins : int, default 200
            Number of initial equal-frequency bins
        fillna : any, default -999999
            Fill value for missing values
        spec_values : list, default []
            List of special values
        random_state : int, default 42
            Random seed
        """
        self.data = data.copy()
        self.original_data = data.copy()
        self.column = column
        self.tgt_name = tgt_name
        self.nbins = nbins
        self.precision = precision
        self.min_bin_prop = min_bin_prop
        self.include_missing = include_missing
        self.equal_freq = equal_freq
        self.bin_colnames = bin_colnames
        self.ascending = ascending
        self.right = right
        self.include_lowest = include_lowest
        self.tree_binning = tree_binning
        self.chi2_method = chi2_method
        self.chi2_p = chi2_p
        self.init_equi_bins = init_equi_bins
        self.fillna = fillna
        self.spec_values = spec_values
        self.random_state = random_state
        self.result = None
        self.bin_edges = None
    
    def run_quick_binning(self):
        """
        Run quick binning.
        
        Run equal-frequency or equal-width binning according to the configured parameters and add the result to the data.
        
        Returns
        -------
        self
            Returns itself to allow method chaining
        """
        if isinstance(self.nbins, int):
            self.nbins = int(get_max_nbins(data=self.data, nbins=self.nbins, 
                                            min_bin_prop=self.min_bin_prop))
        
        if not self.include_missing:
            self.data = self.data.dropna(subset=[self.column])
        
        labels = None
        binned, self.bin_edges = quick_binning(
            data=self.data, 
            column=self.column, 
            nbins=self.nbins, 
            precision=self.precision, 
            equal_freq=self.equal_freq, 
            labels=labels, 
            right=self.right, 
            include_lowest=self.include_lowest,
            tree_binning=self.tree_binning,
            target=self.tgt_name,
            ascending=self.ascending,
            include_missing=self.include_missing,
            random_state=self.random_state,
            spec_values=self.spec_values
        )
        
        left_sign = '[' if self.include_lowest else '('
        right_sign = ']' if self.right else ')'
        bin_range_list = get_bin_range(edges=self.bin_edges, precision=self.precision, 
                                         ascending=self.ascending, left_sign=left_sign, 
                                         right_sign=right_sign)
        
        rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not self.include_missing else [i for i in range(0, len(bin_range_list))]
        binned = binned.cat.rename_categories(rename_catlist)
        
        bin_num_col = self.bin_colnames[0]
        bin_range_col = self.bin_colnames[1]
        
        self.data = _materialize_bin_columns(
            self.data, binned, bin_range_list, bin_num_col, bin_range_col
        )
        
        self.result = self.data
        return self
    
    def run_chi2_binning(self, init_points=None):
        """
        Run chi-square binning.
        
        Building on quick binning, further apply the chi-square test to optimize the bins automatically.
        
        Parameters
        ----------
        init_points : array-like, optional
            Initial bin edges, used as the starting point of chi-square binning
        
        Returns
        -------
        self
            Returns itself to allow method chaining
        """
        self.data = self.original_data.copy()
        self.data = self.data.reset_index(drop=True)
        
        df = self.data[[self.column, self.tgt_name]].copy()
        
        equi_method = "equid"
        if self.equal_freq:
            equi_method = "equif"

        if self.include_missing:
            df[self.column] = df[self.column].fillna(self.fillna)
            spec_values = [self.fillna, *self.spec_values]
        else:
            df = df.dropna()
            self.data = self.data[~pd.isnull(self.data[self.column])]

        nvb = NumVarBinning(var_name=self.column, spec_values=spec_values, 
                             spec_digit=self.precision)  
        binning_series = nvb.auto_binning(
            df=df, 
            tgt_name=self.tgt_name, 
            max_bins=self.nbins, 
            min_prop_in_bin=self.min_bin_prop, 
            equi_method=equi_method,
            equi_bins=self.init_equi_bins, 
            binning_criteria='chi2',
            chi2_p=self.chi2_p,
            init_points=init_points
        )
        
        bin_edges = cat_2_list(binning_series)
        
        left_sign = '['
        right_sign = ')'
        if not self.ascending:
            left_sign = '('
            right_sign = ']'
        bin_range_list = get_bin_range(edges=bin_edges, precision=self.precision, 
                                         ascending=self.ascending, left_sign=left_sign, 
                                         right_sign=right_sign)
        
        binned = binning_series
        rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not self.include_missing else [i for i in range(0, len(bin_range_list))]
        binned = binned.cat.rename_categories(rename_catlist)
        
        bin_num_col = self.bin_colnames[0]
        bin_range_col = self.bin_colnames[1]
        
        self.data = _materialize_bin_columns(
            self.data, binned, bin_range_list, bin_num_col, bin_range_col
        )
        
        self.bin_edges = sorted([np.inf if str(x).lower() == 'inf' else -np.inf if str(x).lower() == '-inf' else x for x in bin_edges])
        
        self.result = self.data
        return self
    
    def run(self):
        """
        Run the binning.
        
        Run either quick binning or chi-square binning, depending on the chi2_method parameter.
        
        Returns
        -------
        self
            Returns itself to allow method chaining
        """
        if self.chi2_method:
            # Run quick binning first to obtain the initial edges
            self.run_quick_binning()
            init_points = self.bin_edges.copy()
            
            # Then run chi-square binning
            self.run_chi2_binning(init_points=init_points)
        else:
            self.run_quick_binning()
        
        return self
    
    def get_result(self, return_edges=True):
        """
        Get the binning result.
        
        Parameters
        ----------
        return_edges : bool, default True
            Whether to return the bin edges
        
        Returns
        -------
        tuple or pandas.DataFrame
            If return_edges is True, return the tuple (result, bin_edges);
            otherwise return only result
        """
        if return_edges:
            return self.result, self.bin_edges
        return self.result


def chi2_binning(data, column, nbins = 10, precision = 5, min_bin_prop = 0.05, tgt_name = None,
                 include_missing = True, equal_freq = True, bin_colnames = ("_bin_num", "_bin_range"), ascending = True, 
                 chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], init_points = None):
    """
    Binning function based on the chi-square test.
    
    Use the chi-square test to find the optimal bin edges automatically: the test decides whether adjacent bins should be merged,
    yielding statistically significant bins in the end.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table
    column : str
        Name of the column to bin
    nbins : int, default 10
        Maximum number of bins
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin
    tgt_name : str
        Target variable name (binary label, 0 and 1)
    include_missing : bool, default True
        Whether to include missing values
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        Tuple of the column names of the binning results
    ascending : bool, default True
        Whether the bin order is ascending
    chi2_p : float, default 0.95
        Significance level of the chi-square test
    init_equi_bins : int, default 100
        Number of initial equal-frequency bins
    fillna : any, default -999999
        Fill value for missing values
    spec_values : list, default []
        List of special values
    init_points : array-like, optional
        Initial bin edges, used as the starting point of chi-square binning
    
    Returns
    -------
    tuple
        (result, bin_edges) - the binning result DataFrame and the array of bin edges
    
    Examples
    --------
    >>> result, edges = chi2_binning(data, column='income', tgt_name='default', nbins=10)
    """
    
    data = data.reset_index(drop = True)
        
    df = data[[column, tgt_name]].copy()
    
    equi_method = "equid"
    if equal_freq:
        equi_method = "equif"

    if include_missing:
        df[column] = df[column].fillna(fillna)
        spec_values = [fillna, *spec_values]
    else:
        df = df.dropna()
        data = data[~pd.isnull(data[column])]

    nvb = NumVarBinning(var_name=column, spec_values=spec_values, spec_digit=precision)  
#     print(init_points)
    binning_series = nvb.auto_binning(
        df=df, 
        tgt_name=tgt_name, 
        max_bins=nbins, 
        min_prop_in_bin=min_bin_prop, 
        equi_method=equi_method,
        equi_bins=init_equi_bins, 
        binning_criteria='chi2',
        chi2_p=chi2_p,
        init_points=init_points)
    
    bin_num_col = bin_colnames[0]
    bin_range_col = bin_colnames[1]
    
    bin_edges = cat_2_list(binning_series)
    if len(bin_edges) < len(binning_series.cat.categories) + 1 and nvb.cut_points is not None:
        # pandas rounds the interval ends of the categories; near the float limits (the -1.797e308 missing sentinel)
        # that overflows to inf and two edges merge, so take the exact cut points instead
        bin_edges = [float(x) for x in nvb.cut_points]
    
    left_sign='['
    right_sign=')'
    if not ascending:
        left_sign='('
        right_sign=']'
    bin_range_list = get_bin_range(edges = bin_edges, precision = precision, ascending = ascending, left_sign=left_sign, right_sign=right_sign)
    
    binned = binning_series
    rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not include_missing else [i for i in range(0, len(bin_range_list))]
    binned = binned.cat.rename_categories(rename_catlist)
    data = _materialize_bin_columns(
        data, binned, bin_range_list, bin_num_col, bin_range_col
    )
    
    fnl_res = data
    
    bin_edges = sorted([np.inf if str(x).lower() == 'inf' else -np.inf if str(x).lower() == '-inf' else x for x in bin_edges])
    
    return fnl_res, bin_edges


def run_binning(data, column, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, equal_freq = True, 
                bin_colnames = ("bin_num", "bin_range"), ascending = False, right = True, include_lowest = False, 
                tree_binning = False, target = None, random_state=42, spec_values = [], fillna = -999999):
    """
    General-purpose binning function supporting equal-frequency or equal-width binning.
    
    Bin a numeric variable with a range of configuration options and return the binned data together with the bin edges.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the column to bin
    nbins : int, default 10
        Number of bins
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin
    include_missing : bool, default True
        Whether to include missing values
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning
    bin_colnames : tuple, default ("bin_num", "bin_range")
        Tuple of the column names of the binning results
    ascending : bool, default False
        Whether the bin order is ascending
    right : bool, default True
        Whether the intervals are closed on the right
    include_lowest : bool, default False
        Whether to include the lowest value
    tree_binning : bool, default False
        Whether to use decision-tree binning
    target : str, optional
        Target variable name (required for decision-tree binning)
    random_state : int, default 42
        Random seed
    spec_values : list, default []
        List of special values
    fillna : scalar, default -999999
        Sentinel value that holds the missing values in their own bin when `include_missing=True`.
    
    Returns
    -------
    tuple
        (data, bin_edges) - the data with the binning columns added and the array of bin edges
    
    Examples
    --------
    >>> data, edges = run_binning(data, column='income', nbins=10, equal_freq=True)
    
    """
    
    # New guard: if the data is empty or the specified column is entirely missing, return a placeholder result directly
    if data.empty or data[column].isnull().all():
        # Return a default result containing a missing-values bin
        bin_num_col, bin_range_col = bin_colnames
        data = data.copy()
        data[bin_num_col] = 0 if include_missing else 1
        data[bin_range_col] = "Missing" if include_missing else "All"
        return data, []
    
    bin_num_col = bin_colnames[0]
    bin_range_col = bin_colnames[1]
    
    if isinstance(nbins, int):
        nbins = int(get_max_nbins(data = data, nbins = nbins, min_bin_prop = min_bin_prop))
    
    if not include_missing:
        data = data.dropna(subset = [column])
    
    labels = None
    binned, bin_edges = quick_binning(data = data, 
                                      column = column, 
                                      nbins = nbins, 
                                      precision = precision, 
                                      equal_freq = equal_freq, 
                                      labels = labels, 
                                      right = right, 
                                      include_lowest = include_lowest,
                                      tree_binning = tree_binning,
                                      target = target,
                                      ascending = ascending,
                                      include_missing = include_missing,
                                      random_state = random_state,
                                      spec_values = spec_values,
                                      fillna = fillna)
    
    left_sign='[' if include_lowest else '('
    right_sign=']' if right else ')'
    bin_range_list = get_bin_range(edges = bin_edges, precision = precision, ascending = ascending, left_sign=left_sign, right_sign=right_sign)
    
    rename_catlist = [i for i in range(1, len(bin_range_list) + 1)] if not include_missing else [i for i in range(0, len(bin_range_list))]
    binned = binned.cat.rename_categories(rename_catlist)
    data = _materialize_bin_columns(
        data, binned, bin_range_list, bin_num_col, bin_range_col
    )
        
    return data, bin_edges


def super_binning(data, score, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, 
                  equal_freq = True, chi2_method = False, chi2_p = 0.95, init_equi_bins = 2000, fillna = -999999, 
                  spec_values = [], tree_binning = False, random_state=42, return_edges = False, ascending = True,
                  bin_colnames = ("_bin_num", "_bin_range")):
    """
    Super binning function that combines several binning strategies.
    
    Provide a unified binning interface with two modes, basic binning and chi-square binning,
    and automatically choose a suitable binning strategy from the parameter configuration.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Input data table
    score : str
        Name of the score/numeric column to bin
    dep : str
        Target variable name (binary label, 0 and 1)
    nbins : int, default 10
        Maximum number of bins
    precision : int, default 5
        Precision of the edge values (number of decimal places)
    min_bin_prop : float, default 0.05
        Minimum proportion of samples per bin
    include_missing : bool, default True
        Whether to include missing values
    equal_freq : bool, default True
        True for equal-frequency binning, False for equal-width binning
    chi2_method : bool, default False
        Whether to refine the bins with chi-square binning
    chi2_p : float, default 0.95
        Significance level of the chi-square test
    init_equi_bins : int, default 2000
        Number of initial equal-frequency bins (before chi-square binning)
    fillna : any, default -999999
        Fill value for missing values
    spec_values : list, default []
        List of special values
    tree_binning : bool, default False
        Whether to use decision-tree binning
    random_state : int, default 42
        Random seed
    return_edges : bool, default False
        Whether to return the bin edges
    ascending : bool, default True
        Whether the bin order is ascending
    bin_colnames : tuple, default ("_bin_num", "_bin_range")
        Tuple of the column names of the binning results
    
    Returns
    -------
    pandas.DataFrame or tuple
        If return_edges is False, return the binning result data
        If return_edges is True, return the tuple (result, edges)
    
    Examples
    --------
    >>> # Basic binning
    >>> result = super_binning(data, score='income', dep='default', nbins=10)
    
    >>> # Chi-square binning
    >>> result, edges = super_binning(data, score='income', dep='default', 
    ...                                nbins=10, chi2_method=True, return_edges=True)
    """
    
    res, output_edges = run_binning(data = data, 
                                    column = score, 
                                    nbins = nbins, 
                                    precision = precision, 
                                    min_bin_prop = min_bin_prop,
                                    include_missing = include_missing, 
                                    equal_freq = equal_freq,
                                    bin_colnames = bin_colnames,
                                    ascending = ascending,
                                    tree_binning = tree_binning,
                                    target = dep,
                                    random_state = random_state,
                                    spec_values = spec_values,
                                    fillna = fillna)
    
#     print("First Layer Edges: ", output_edges)
    
    if chi2_method:
        """ Chi2 Binning. """
        
#         chi2_edges = [x for x in output_edges if x not in spec_values + [-np.inf, np.inf]]
        chi2_edges = [x for x in output_edges if x not in [-np.inf, np.inf]]
        
#         print("Special Value: ", spec_values)
#         print("Second Layer Inputs: ", chi2_edges)
        res, output_edges = chi2_binning(data = data, 
                                         column = score, 
                                         tgt_name = dep,
                                         nbins = nbins, 
                                         precision = precision, 
                                         min_bin_prop = min_bin_prop, 
                                         include_missing = include_missing, 
                                         equal_freq = equal_freq, 
                                         bin_colnames = bin_colnames, 
                                         ascending = ascending, 
                                         chi2_p = chi2_p, 
                                         init_equi_bins = init_equi_bins, 
                                         fillna = fillna, 
                                         spec_values = spec_values,
                                         init_points = chi2_edges)
        
        
    if return_edges:
        return res, output_edges
    
    return res
