import logging

import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np


logging.basicConfig(level=logging.INFO, format="%(message)s")

class kSeries(Series):
    """
    The extension class of Pandas Series, paired with ``kDataFrame``: operations on it return ``kSeries`` /
    ``kDataFrame`` objects, and it adds a few scoring and frequency-table helpers.

    Parameters
    ----------
    data : array-like, Iterable, dict or scalar value
        The data of the series, exactly as accepted by ``pandas.Series``. The argument is required.
    *args : tuple
        Further positional arguments, forwarded unchanged to ``pandas.Series``.
    **kwargs : dict
        Further keyword arguments (for example ``index``, ``dtype``, ``name`` and ``copy``), forwarded unchanged to
        ``pandas.Series``.
    """
    @property
    def _constructor(self):
        return kSeries

    @property
    def _constructor_expanddim(self):
        return kDataFrame
    
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
    def to_pdSeries(self, inplace: bool = False):
        """
        Convert kSeries back to pandas Series.

        Parameters
        ----------
        inplace : bool, default False
            If False, convert a copy of the kSeries. If True, convert the kSeries itself without copying it first, so
            the returned Series shares its data with this object. The kSeries is never modified or replaced.

        Returns
        -------
        pandas.Series
            A plain ``pandas.Series`` (not a ``kSeries``) with the same values, index and name.
        """
        if inplace:
            data = self
        else:
            data = self.copy()

        df = Series(data)
        return df
    
    def odds_score(self):
        """
        Calculate the odd score based on the given probability score.

        Odds Score = (count of sth happening) / (count of sth not happening)
        Odds Score = pb_score / (1-pb_score)  [This ranges from 0 to infinity.]
        Log Odds Score = np.log(pb_score / (1-pb_score))  [This ranges from -infinity to +infinity]
        This is helpful for solving binary classification problem.
        Odds Ratio: The ratio of odds.

        The values of the series are used as ``pb_score``, the predicted probability of the event.

        Returns
        -------
        kSeries
            The scorecard-style score ``500 - 20 / ln(2) * (ln(15) + ln(pb_score / (1 - pb_score)))`` for every value,
            with the index and name of the series. It is not the raw odds: a higher probability gives a lower score,
            every doubling of the odds lowers the score by 20 points, and ``pb_score = 1/16`` (odds of 1 to 15) maps
            to exactly 500.

        Notes
        -----
        The probabilities are not validated. ``0`` gives ``inf``, ``1`` gives ``-inf``, and values outside ``[0, 1]``
        or missing values give NaN. For other base score, odds or points-to-double-the-odds settings use
        ``Modeling_Tool.Core.utils.odds_score``.
        """
        pb_score = self
        a = (20 / np.log(2))
        b = (np.log(15) + np.log(pb_score/(1 - pb_score)))
        return (500 - a * b)
    
    def scale_score(self):
        """
        Scale the model scores (for internt segment of MCI model)

        Returns
        -------
        kSeries
            The values multiplied by 1.112 and capped at 0.9999999 (the index and name of the series are kept).
        """
        return (self * 1.112).clip(upper=0.9999999)
    
    def proc_freq(self) -> pd.DataFrame:
        """
        Implement the Python version "proc freq" query in SAS.

        Returns
        -------
        pandas.DataFrame
            One row per distinct value of the series (missing values count as a row of their own), indexed by that
            value and sorted by decreasing frequency. The columns are ``frequency`` (count), ``percent`` (share of all
            values), ``cumFrequency`` and ``cumPercent`` (running sums of the first two, in the row order).
        """
        data = self.copy()
        f = data.value_counts(dropna = False)
        p = data.value_counts(dropna = False, normalize = True)
        df = pd.concat([f,p], axis = 1, keys = ['frequency', 'percent'])
        df['cumFrequency'] = df['frequency'].cumsum()
        df['cumPercent'] = df['percent'].cumsum()
        return df
    
    
        
class kDataFrame(DataFrame):
    """
    The extension class of Pandas DataFrame including more useful methods to manipulate dataset.

    Parameters
    ----------
    data : DataFrame
        Any pandas dataframe dataset. Anything accepted by ``pandas.DataFrame`` (dict, list of records, ndarray,
        DataFrame, ...) works, but unlike pandas the argument is required.
    *args : tuple
        Further positional arguments, forwarded unchanged to ``pandas.DataFrame``.
    **kwargs : dict
        Further keyword arguments (for example ``index``, ``columns``, ``dtype`` and ``copy``), forwarded unchanged to
        ``pandas.DataFrame``.

    Returns
    -------
    kDataFrame
        The extension object of dataframe.

    Attributes
    ----------
    added_property : int, default 1
        Example of a metadata attribute: it is listed in ``_metadata``, so pandas passes it on to copies. SMF does
        not use it anywhere else.
    """
    _metadata = ['added_property']
    added_property = 1  # This will be passed to copies

    @property
    def _constructor(self):
        return kDataFrame

    @property
    def _constructor_sliced(self):
        return kSeries
    
    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        
    def move_column(self, colname: str, idx: int, inplace: bool = False):
        """
        To move a column into specific place by index.

        Parameters
        ----------
        colname : str
            Name of the column to move. It must be a column of the frame.
        idx : int
            Target position (0-based) of the column, with ``list.insert`` semantics: 0 puts it first and a value
            at or beyond the number of columns puts it last.
        inplace : bool, default False
            If False, work on a copy; if True, work on the frame itself. This does not change the result: the
            reordered frame is built with ``data[colarray]``, a new object in both cases, so the frame the method is
            called on is never reordered (see Notes).

        Returns
        -------
        kDataFrame
            A new frame with the same data and the columns reordered.

        Notes
        -----
        ``inplace=True`` is effectively a no-op for the original frame: always use the returned frame.
        """
        if inplace:
            data = self
        else:
            data = self.copy()
        colarray = data.columns.tolist()
        colarray.remove(colname)
        colarray.insert(idx, colname)
        data = data[colarray]
        return data

    def convert_to_vintage(self, vintage_colname: str = 'VINTAGE', by: str = 'TRAN_TMS', inplace: bool = False):
        """
        To obtain a vintage column by a time/data column.

        Parameters
        ----------
        vintage_colname : str, default 'VINTAGE'
            Name of the column that receives the vintage. A column with this name is overwritten.
        by : str, default 'TRAN_TMS'
            Name of the date/time column the vintage is taken from. The column is cast to string and the first
            ``YYYY-MM`` pattern in the text is used (for example from ``2024-05-17`` or ``2024-05-17 10:30:00``).
        inplace : bool, default False
            If True, add the column to the frame itself; if False, add it to a copy and leave the frame unchanged.

        Returns
        -------
        kDataFrame
            The frame holding the new column: the frame itself when ``inplace=True``, otherwise a copy. The column
            has pandas ``string`` dtype and holds the vintage as ``YYYYMM`` text (``'202405'``).

        Notes
        -----
        Values that contain no ``YYYY-MM`` pattern (other separators such as ``2024/05/17``, compact dates such as
        ``20240517``, or missing values) become missing (``<NA>``) in the vintage column.
        """
        if inplace:
            data = self
        else:
            data = self.copy()
        data[vintage_colname] = (
            data[by]
            .astype("string")
            .str.extract(r"(\d{4}-\d{2})", expand=False)
            .str.replace("-", "", regex=False)
        )
        if inplace:
            self = data
        return data

    def col_filter_regex(self, regex: str = ".*?of_co_at_12m", case_sensitive = True, inplace: bool = False):
        """
        To filter the DataFrame columns by regular expression.

        Parameters
        ----------
        regex : str, default ".*?of_co_at_12m"
            Regular expression searched (``re.search``, so it may match anywhere) in every column name; the columns
            whose name contains a match are kept.
        case_sensitive : bool, default True
            If False, the match ignores case.
        inplace : bool, default False
            If False, work on a copy; if True, work on the frame itself. This does not change the result: the frame
            the method is called on is never modified in either case.

        Returns
        -------
        kDataFrame
            A new frame holding only the matching columns, in their original order (a frame without columns when
            nothing matches).

        Notes
        -----
        The column names must be strings, because the pattern is applied through ``DataFrame.columns.str``.
        """
        if inplace:
            data = self
        else:
            data = self.copy()

        fltr = data.columns[data.columns.str.contains(regex, regex = True, case = case_sensitive)]
        return data[fltr]
    
    def row_filter_regex(self, col = None, regex: str = None, case_sensitive = True,
                         as_index = False, inplace: bool = False):
        """
        To filter the string format row using regex.

        Parameters
        ----------
        col : str, default None
            Name of the column whose values are tested. Required in practice: ``None`` raises ``KeyError``.
        regex : str, default None
            Regular expression searched (``re.search``, so it may match anywhere) in the text of every value.
            Required in practice: ``None`` raises ``TypeError``.
        case_sensitive : bool, default True
            If False, the match ignores case.
        as_index : bool, default False
            If True, the returned rows are indexed by ``col`` (``set_index(col)``), so the column is moved out of the
            regular columns.
        inplace : bool, default False
            If False, work on a copy; if True, work on the frame itself. This does not change the result: the frame
            the method is called on is never modified in either case.

        Returns
        -------
        kDataFrame
            A new frame with the rows whose value in ``col`` matches ``regex``, keeping their original index (or
            indexed by ``col`` when ``as_index=True``).

        Notes
        -----
        The values of ``col`` are cast to ``str`` before matching, so numbers are matched through their text
        (``'1.5'``) and missing values through ``'nan'`` or ``'None'``.
        """
        if inplace:
            data = self
        else:
            data = self.copy()
            
        fltr = data[col].astype('str').str.contains(pat = regex, regex = True, case = case_sensitive)
        if as_index:
            return data[fltr].set_index(col)
        return data[fltr]
    
    def scale_score(self, pb_score: str):
        """
        Scale the model scores (for internt segment of MCI model)

        Parameters
        ----------
        pb_score : str
            Name of the column that holds the model score (a probability).

        Returns
        -------
        kSeries
            The column multiplied by 1.112 and capped at 0.9999999, with the index of the frame. The frame itself
            is not modified (no column is added).
        """
        return (self[pb_score] * 1.112).clip(upper=0.9999999)
    
    def proc_freq(self, var: str):
        """
        Implement the Python version "proc freq" query in SAS.

        Parameters
        ----------
        var : str
            Name of the column to tabulate.

        Returns
        -------
        pandas.DataFrame
            One row per distinct value of the column (missing values count as a row of their own), indexed by that
            value and sorted by decreasing frequency. The columns are ``frequency`` (count), ``percent`` (share of all
            rows), ``cumFrequency`` and ``cumPercent`` (running sums of the first two, in the row order).
        """
        data = self.copy()
        f = data[var].value_counts(dropna = False)
        p = data[var].value_counts(dropna = False, normalize = True)
        df = pd.concat([f,p], axis = 1, keys = ['frequency', 'percent'])
        df['cumFrequency'] = df['frequency'].cumsum()
        df['cumPercent'] = df['percent'].cumsum()
        return df
    
    def unify_table_col_names(self, how: str = "lowercase", inplace: bool = False):
        """
        Unify the format of column names.

        Parameters
        ----------
        how : str, default "lowercase"
            Target format, compared case-insensitively: ``"lower"`` or ``"lowercase"``, ``"upper"`` or
            ``"uppercase"``, ``"cap"`` or ``"capitalize"`` (first letter upper case, the rest lower case).
        inplace : bool, default False
            If True, rename the columns of the frame itself; if False, work on a copy and leave the frame unchanged.

        Returns
        -------
        kDataFrame
            The frame with the renamed columns: the frame itself when ``inplace=True``, otherwise a copy.

        Notes
        -----
        The column names must be strings. ``how`` is not validated: any other value fails with
        ``UnboundLocalError``. Names that collide after the conversion (``'A'`` and ``'a'``) become duplicate
        column names.
        """
        
        if inplace:
            data = self
        else:
            data = self.copy()
        
        cols = data.columns
        if how.lower() == "lower" or how.lower() == "lowercase":
            res = [name.lower() for name in cols]
        if how.lower() == "upper" or how.lower() == "uppercase":
            res = [name.upper() for name in cols]
        if how.lower() == "cap" or how.lower() == "capitalize":
            res = [name.capitalize() for name in cols]
        data.columns = res
        return data
    
    def convert_strlist_to_list(self, col: str):
        """
        cast string-type lists in a specified Series into real lists.

        Parameters
        ----------
        col : str
            Name of the column that holds the list-like strings, for example ``"['a', 'b']"``.

        Returns
        -------
        kSeries
            One list of strings per row: the runs of letters, digits and underscores found in ``str(value)``, so
            quotes, brackets and commas are dropped (``"['a', 'b']"`` becomes ``['a', 'b']``).

        Notes
        -----
        The series is rebuilt from the extracted lists, so it has a new default index (0 to n-1) and not the index
        of the frame. Any other punctuation also splits a token (``'1.5'`` becomes ``['1', '5']``), and a missing
        value becomes ``['nan']`` or ``['None']``.
        """
        import re
        
        data = self.copy()
        str_col = kSeries([re.findall("\w+", str(x)) for x in data[col]])
        return str_col
    
    def to_pdDataFrame(self, inplace: bool = False):
        """
        Convert df_extension back to DataFrame.

        Parameters
        ----------
        inplace : bool, default False
            If False, convert a copy of the kDataFrame. If True, convert the kDataFrame itself without copying it
            first, so the returned DataFrame shares its data with this object. The kDataFrame is never modified or
            replaced.

        Returns
        -------
        pandas.DataFrame
            A plain ``pandas.DataFrame`` (not a ``kDataFrame``) with the same data, index and columns.
        """
        if inplace:
            data = self
        else:
            data = self.copy()
            
        df = DataFrame(data)
        return df
