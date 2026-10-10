# =============================================================================
# Modeling_Tool.Model.Backward_Tool
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-BACKWARDTOOL-11b388c0
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import os
import sys
import copy
import logging
import functools
from collections import OrderedDict
from typing import Optional, List, Dict, Union, Any, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from Modeling_Tool.Core.sample_weight_utils import resolve_sample_weight
def _resolve_backward_perf_weight_col(split_name: str, df: pd.DataFrame, weight_col: Optional[str], validation_weight_col: Optional[str]) -> Optional[str]: ...
def _backward_perf_summary(df_score: pd.DataFrame, *, split_name: str, dep: str, score_col: str, weight_col: Optional[str] = None, validation_weight_col: Optional[str] = None, nbins: int = 10, precision: int = 5, min_bin_prop: float = 0.05, include_missing: bool = True, equal_freq: bool = True): ...
def backward_lgbm(train_data: pd.DataFrame, varlist: List[str], dep: str, varreduct_params: Optional[Dict] = None, stopping_metric: str = 'auc', seed: int = 42, num_boost_round: int = 200, early_stopping_rounds: int = 20, importance_type: str = 'gain', cum_importance_threshold: float = 0.99, min_vars: int = 10, validation_data: Optional[pd.DataFrame] = None, test_data_dict: Optional[Dict[str, pd.DataFrame]] = None, ret_perf: bool = True, nbins: int = 10, precision: int = 5, min_bin_prop: float = 0.05, include_missing: bool = True, equal_freq: bool = True, ascending: bool = True, fillna: Optional[float] = None, spec_values: Optional[List] = None, weight_col: Optional[str] = None, validation_weight_col: Optional[str] = None, wgt_col: Optional[str] = None) -> Tuple: ...
def backward_xgbm(train_data: pd.DataFrame, varlist: List[str], dep: str, varreduct_params: Optional[Dict] = None, stopping_metric: str = 'auc', seed: int = 42, num_boost_round: int = 200, early_stopping_rounds: int = 20, importance_type: str = 'gain', cum_importance_threshold: float = 0.99, min_vars: int = 10, validation_data: Optional[pd.DataFrame] = None, test_data_dict: Optional[Dict[str, pd.DataFrame]] = None, ret_perf: bool = True, nbins: int = 10, precision: int = 5, min_bin_prop: float = 0.05, include_missing: bool = True, equal_freq: bool = True, ascending: bool = True, fillna: Optional[float] = None, spec_values: Optional[List] = None, monotone_constraints: Optional[Dict[str, int]] = None, weight_col: Optional[str] = None, validation_weight_col: Optional[str] = None, wgt_col: Optional[str] = None) -> Tuple: ...

class BackwardVariableEliminator:
    def __init__(self, train_data: pd.DataFrame, varlist: List[str], dep: str, model_type: str = 'lgbm', validation_data: Optional[pd.DataFrame] = None, test_data_dict: Optional[Dict[str, pd.DataFrame]] = None, weight_col: Optional[str] = None, validation_weight_col: Optional[str] = None, wgt_col: Optional[str] = None): ...
    def run(self, n_rounds: int = 5, varreduct_params: Optional[Dict] = None, stopping_metric: str = 'auc', seed: int = 42, num_boost_round: int = 200, early_stopping_rounds: int = 20, importance_type: str = 'gain', cum_importance_threshold: float = 0.99, min_vars: int = 10, ret_perf: bool = True, nbins: int = 10, **kwargs) -> List[Dict]: ...
    def get_final_vars(self) -> List[str]: ...
    def get_summary(self) -> pd.DataFrame: ...

class BackwardEliminationAnalyzer:
    def __init__(self, results: List[Dict]): ...
    def get_stable_vars(self, top_n: Optional[int] = None) -> List[str]: ...
    def plot_var_reduction(self, figsize: Tuple[int, int] = (8, 4), save_path: Optional[str] = None) -> None: ...
    def get_perf_trend(self, dataset: str = 'mdl', metric: str = 'IV') -> pd.DataFrame: ...
