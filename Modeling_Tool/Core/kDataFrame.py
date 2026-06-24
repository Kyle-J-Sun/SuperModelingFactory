import logging

import pandas as pd
from pandas import DataFrame
from pandas import Series
import numpy as np

# Available only in newer pandas versions. Older Airflow images should skip it.
try:
    pd.set_option('future.no_silent_downcasting', True)
except (KeyError, ValueError):
    pass
pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.INFO, format="%(message)s")

class kSeries(Series):
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
        """
        pb_score = self
        a = (20 / np.log(2))
        b = (np.log(15) + np.log(pb_score/(1 - pb_score)))
        return (500 - a * b)
    
    def scale_score(self):
        """
        Scale the model scores (for internt segment of MCI model)
        """
        data = self.copy()
        def app_func(x):
            scores = x * 1.112
            if (scores > 0.9999999):
                return 0.9999999
            return scores
        return data.apply(app_func)
    
    def proc_freq(self) -> pd.DataFrame:
        """
        Implement the Python version "proc freq" query in SAS.
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
    
    Parameters:
    -----------
    data [DataFrame]: Any pandas dataframe dataset.
    
    Return:
    -------
    data [kDataFrame]: the extension object of dataframe.
    
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
        """
        if inplace:
            data = self
        else:
            data = self.copy()
        import re
        data[vintage_colname] = data[by].apply(lambda x: re.search("\d{4}-\d{2}", x).group().replace('-', ''))
        if inplace:
            self = data
        return data

    def col_filter_regex(self, regex: str = ".*?of_co_at_12m", case_sensitive = True, inplace: bool = False):
        """
        To filter the DataFrame columns by regular expression.
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
        """
        data = self.copy()
        def app_func(x):
            scores = x * 1.112
            if (scores > 0.9999999):
                return 0.9999999
            return scores
        return data[pb_score].apply(app_func)
    
    def proc_freq(self, var: str):
        """
        Implement the Python version "proc freq" query in SAS.
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
        """
        import re
        
        data = self.copy()
        str_col = kSeries([re.findall("\w+", str(x)) for x in data[col]])
        return str_col
    
    def to_pdDataFrame(self, inplace: bool = False):
        """
        Convert df_extension back to DataFrame.
        """
        if inplace:
            data = self
        else:
            data = self.copy()
            
        df = DataFrame(data)
        return df
