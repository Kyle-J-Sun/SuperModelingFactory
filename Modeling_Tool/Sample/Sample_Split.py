"""
Sampling classes for credit modeling.

This module provides classes for splitting samples, stratified sampling,
and sample balancing for credit model development.

Classes
-------
SampleSplitter : Split data into train/test samples.
StratifiedSampler : Stratified sampling with target balance control.
SampleBalancer : Balance samples using various techniques.

Examples
--------
>>> from Modeling_Tool_refactored.sample import SampleSplitter
>>> splitter = SampleSplitter()
>>> train, test = splitter.split(df, 'target', test_size=0.3)
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from Modeling_Tool.Eval.Model_Eval_Tool import PerformanceEvaluator
from Modeling_Tool.Core.utils import get_feature_names

def select_sample_seed(master_df, oot_split_col, model, tgt_name, seed_range = (3000, 3050), ins_prop = 0.7):
    """ Select Best Seed for Sample Splitting. """
    
    from tqdm import tqdm
    
    if isinstance(model, str):
        score = model

    perf_res = pd.DataFrame()
    for seed in tqdm(range(seed_range[0], seed_range[1])):

        train_df = master_df.loc[master_df[oot_split_col].isin([1])]
        oot_df = master_df.loc[master_df[oot_split_col].isin([2])]

        sampler = SampleSplitter(test_size = (1 - ins_prop), random_state=seed, stratify=True)
        mdl_df, val_df = sampler.split_df(train_df, tgt_name)

        mdl_df['sample_ind_fnl'] = "ins"
        val_df['sample_ind_fnl'] = "oos"
        oot_df['sample_ind_fnl'] = "oot"

        drv_w_sample_ind = pd.concat([mdl_df, val_df, oot_df])

        ins_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['ins'])]
        oos_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['oos'])]
        oot_df = drv_w_sample_ind.loc[drv_w_sample_ind['sample_ind_fnl'].isin(['oot'])]

        evaluator = PerformanceEvaluator(tgt_name=tgt_name, model=model, feature_cols=get_feature_names(model)) if not isinstance(model, str) else PerformanceEvaluator(tgt_name=tgt_name, scr_name = score)
        
        if ins_df.shape[0] > 0:
            evaluator = evaluator.add_dataset('train', ins_df)
            
        if oos_df.shape[0] > 0:
            evaluator = evaluator.add_dataset('validation', oos_df)
            
        if oot_df.shape[0] > 0:
            evaluator = evaluator.add_dataset('oot', oot_df)
            
        result = evaluator.evaluate(to_show=False, display = False)
        result['seed'] = seed
        
        perf_res = pd.concat([perf_res, result], axis = 0)
        
    return perf_res


class SampleSplitter:
    """
    Split data into training and testing samples.
    
    This class provides flexible sample splitting with support for
    stratification, random sampling, and custom split ratios.
    
    Parameters
    ----------
    test_size : float, default 0.3
        Proportion of data for testing (0 to 1).
    random_state : int, optional
        Random seed for reproducibility.
    stratify : bool, default True
        Whether to stratify by target variable.
    
    Attributes
    ----------
    train_index_ : numpy.ndarray
        Indices for training data.
    test_index_ : numpy.ndarray
        Indices for testing data.
    
    Methods
    -------
    split(X, y, test_size=None, stratify=None)
        Split data into train and test sets.
    split_df(df, target, exclude_cols=None)
        Split DataFrame while excluding certain columns.
    
    Examples
    --------
    >>> splitter = SampleSplitter(test_size=0.2, random_state=42)
    >>> train, test = splitter.split(X, y)
    >>> print(f"Train size: {len(train)}, Test size: {len(test)}")
    """
    
    def __init__(self, test_size: float = 0.3, 
                 random_state: Optional[int] = None,
                 stratify: bool = True):
        """
        Initialize SampleSplitter.
        
        Parameters
        ----------
        test_size : float, default 0.3
            Proportion for testing.
        random_state : int, optional
            Random seed.
        stratify : bool, default True
            Whether to stratify.
        """
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.train_index_ = None
        self.test_index_ = None
    
    def split(self, X: Union[pd.DataFrame, np.ndarray],
              y: Union[pd.Series, np.ndarray],
              test_size: Optional[float] = None,
              stratify: Optional[bool] = None) -> Tuple:
        """
        Split data into train and test sets.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features.
        y : pandas.Series or numpy.ndarray
            Target variable.
        test_size : float, optional
            Override default test size.
        stratify : bool, optional
            Override default stratify setting.
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        
        Examples
        --------
        >>> X_train, X_test, y_train, y_test = splitter.split(X, y)
        """
        test_size = test_size if test_size is not None else self.test_size
        stratify = stratify if stratify is not None else self.stratify
        
        y_arr = np.array(y)
        stratify_param = y_arr if stratify and len(np.unique(y_arr)) > 1 else None
        
        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )
    
    def split_df(self, df: pd.DataFrame, target: str,
                exclude_cols: Optional[List[str]] = None,
                test_size: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split DataFrame while excluding certain columns from split.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input DataFrame.
        target : str
            Target column name.
        exclude_cols : list of str, optional
            Columns to exclude from split.
        test_size : float, optional
            Override default test size.
        
        Returns
        -------
        tuple
            (train_df, test_df)
        
        Examples
        --------
        >>> train_df, test_df = splitter.split_df(df, 'target', exclude_cols=['id', 'date'])
        """
        test_size = test_size if test_size is not None else self.test_size
        
        exclude_cols = exclude_cols or []
        feature_cols = [c for c in df.columns if c not in exclude_cols + [target]]
        
        X = df[feature_cols]
        y = df[target]
        
        if self.stratify:
            strat = y
        else:
            strat = None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=strat
        )
        
        train_df = X_train.copy()
        train_df[target] = y_train
        
        test_df = X_test.copy()
        test_df[target] = y_test
        
        return train_df, test_df


class StratifiedSampler:
    """
    Stratified sampling with target balance control.
    
    This class provides stratified sampling that maintains
    the target distribution while allowing controlled sampling.
    
    Parameters
    ----------
    target_rate : float, optional
        Target bad rate in the sample.
    random_state : int, optional
        Random seed.
    
    Methods
    -------
    sample(df, target, n_samples=None, sample_frac=None)
        Perform stratified sampling.
    balance(df, target, method='undersample')
        Balance sample by adjusting target distribution.
    
    Examples
    --------
    >>> sampler = StratifiedSampler(target_rate=0.15)
    >>> balanced = sampler.balance(df, 'target', method='undersample')
    """
    
    def __init__(self, target_rate: Optional[float] = None,
                 random_state: Optional[int] = None):
        """
        Initialize StratifiedSampler.
        """
        self.target_rate = target_rate
        self.random_state = random_state
        self.original_rate_ = None
    
    def sample(self, df: pd.DataFrame, target: str,
              n_samples: Optional[int] = None,
              sample_frac: Optional[float] = None) -> pd.DataFrame:
        """
        Perform stratified sampling.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data.
        target : str
            Target column name.
        n_samples : int, optional
            Number of samples to draw.
        sample_frac : float, optional
            Fraction of data to sample.
        
        Returns
        -------
        pandas.DataFrame
            Sampled DataFrame.
        
        Examples
        --------
        >>> sampled = sampler.sample(df, 'target', sample_frac=0.5)
        """
        if sample_frac is not None:
            return df.sample(frac=sample_frac, random_state=self.random_state)
        
        if n_samples is not None:
            return df.sample(n=n_samples, random_state=self.random_state)
        
        return df.copy()
    
    def balance(self, df: pd.DataFrame, target: str,
               method: str = 'undersample') -> pd.DataFrame:
        """
        Balance sample by adjusting target distribution.
        
        Parameters
        ----------
        df : pandas.DataFrame
            Input data.
        target : str
            Target column name.
        method : str, default 'undersample'
            Balancing method: 'undersample', 'oversample', 'smote'.
        
        Returns
        -------
        pandas.DataFrame
            Balanced DataFrame.
        
        Examples
        --------
        >>> balanced = sampler.balance(df, 'target', method='undersample')
        """
        self.original_rate_ = df[target].mean()
        
        if method == 'undersample':
            return self._undersample(df, target)
        elif method == 'oversample':
            return self._oversample(df, target)
        elif method == 'smote':
            return self._smote(df, target)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _undersample(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Undersample majority class.
        """
        goods = df[df[target] == 0]
        bads = df[df[target] == 1]
        
        if self.target_rate is not None:
            n_bads = len(bads)
            n_goods = int(n_bads * (1 - self.target_rate) / self.target_rate)
            goods = goods.sample(n=min(n_goods, len(goods)), random_state=self.random_state)
        else:
            goods = goods.sample(n=len(bads), random_state=self.random_state)
        
        return pd.concat([goods, bads]).sample(frac=1, random_state=self.random_state)
    
    def _oversample(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Oversample minority class.
        """
        goods = df[df[target] == 0]
        bads = df[df[target] == 1]
        
        if len(bads) < len(goods):
            if self.target_rate is not None:
                n_goods = len(goods)
                n_bads = int(n_goods * self.target_rate / (1 - self.target_rate))
                bads = bads.sample(n=n_bads, replace=True, random_state=self.random_state)
            else:
                bads = bads.sample(n=len(goods), replace=True, random_state=self.random_state)
        
        return pd.concat([goods, bads]).sample(frac=1, random_state=self.random_state)
    
    def _smote(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        SMOTE oversampling (requires imbalanced-learn).
        """
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("imbalanced-learn required for SMOTE. Install with: pip install imbalanced-learn")
        
        goods = df[df[target] == 0]
        bads = df[df[target] == 1]
        
        if len(bads) < 6:
            return self._oversample(df, target)
        
        X_cols = [c for c in df.columns if c != target]
        X = df[X_cols].values
        y = df[target].values
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        return pd.DataFrame(X_resampled, columns=X_cols).assign(**{target: y_resampled})


class SampleBalancer:
    """
    Advanced sample balancing with multiple methods.
    
    This class provides various sampling techniques to handle
    class imbalance in credit modeling.
    
    Parameters
    ----------
    method : str, default 'random'
        Balancing method.
    target_ratio : float, optional
        Desired minority/majority ratio.
    random_state : int, optional
        Random seed.
    
    Methods
    -------
    fit_resample(X, y)
        Resample features and target.
    get_balanced_indices(y)
        Get indices for balanced sampling.
    
    Examples
    --------
    >>> balancer = SampleBalancer(method='nearmiss')
    >>> X_bal, y_bal = balancer.fit_resample(X, y)
    """
    
    def __init__(self, method: str = 'random',
                 target_ratio: Optional[float] = None,
                 random_state: Optional[int] = None):
        """
        Initialize SampleBalancer.
        """
        self.method = method
        self.target_ratio = target_ratio
        self.random_state = random_state
    
    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray],
                    y: Union[pd.Series, np.ndarray]) -> Tuple:
        """
        Resample data to balance classes.
        
        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Features.
        y : pandas.Series or numpy.ndarray
            Target.
        
        Returns
        -------
        tuple
            (X_resampled, y_resampled)
        
        Examples
        --------
        >>> X_bal, y_bal = balancer.fit_resample(X, y)
        """
        if self.method == 'random':
            return self._random_undersample(X, y)
        elif self.method == 'nearmiss':
            return self._nearmiss(X, y)
        elif self.method == 'tomek':
            return self._tomek_links(X, y)
        elif self.method == 'enn':
            return self._edited_nn(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _random_undersample(self, X, y):
        """
        Random undersampling.
        """
        goods = y == 0
        bads = y == 1
        
        n_bads = goods.sum() if self.target_ratio else bads.sum()
        
        if self.target_ratio:
            n_goods = int(n_bads * (1 - self.target_ratio) / self.target_ratio)
        else:
            n_goods = n_bads
        
        goods_indices = np.where(goods)[0]
        bads_indices = np.where(bads)[0]
        
        np.random.seed(self.random_state)
        goods_sample = np.random.choice(goods_indices, size=n_goods, replace=False)
        
        selected = np.concatenate([goods_sample, bads_indices])
        np.random.shuffle(selected)
        
        if isinstance(X, pd.DataFrame):
            return X.iloc[selected].copy(), y.iloc[selected].copy()
        return X[selected], y[selected]
    
    def _nearmiss(self, X, y):
        """
        NearMiss undersampling.
        """
        try:
            from imblearn.under_sampling import NearMiss
        except ImportError:
            raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
        
        nm = NearMiss(version=1, random_state=self.random_state)
        return nm.fit_resample(X, y)
    
    def _tomek_links(self, X, y):
        """
        Tomek links cleaning.
        """
        try:
            from imblearn.under_sampling import TomekLinks
        except ImportError:
            raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
        
        tl = TomekLinks(random_state=self.random_state)
        return tl.fit_resample(X, y)
    
    def _edited_nn(self, X, y):
        """
        Edited Nearest Neighbors cleaning.
        """
        try:
            from imblearn.under_sampling import EditedNearestNeighbours
        except ImportError:
            raise ImportError("imbalanced-learn required. Install with: pip install imbalanced-learn")
        
        enn = EditedNearestNeighbours(random_state=self.random_state)
        return enn.fit_resample(X, y)
