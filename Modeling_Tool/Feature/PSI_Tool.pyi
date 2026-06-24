# =============================================================================
# Modeling_Tool.Feature.PSI_Tool
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-PSITOOL-5299b9cc
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Optional, Tuple, Callable, Any
from tqdm import tqdm
from Modeling_Tool.Core.Binning_Tool import quick_binning

class PSICalculator:
    def __init__(self, buckets: int = 10, equal_freq: bool = True, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5): ...
    def calculate(self, expected_df: pd.DataFrame, current_data: pd.DataFrame, varlist: List[str], group_by: Optional[str] = None, group_name: Optional[str] = None, return_details = False) -> pd.DataFrame: ...
def calculate_psi(expected: Union[pd.DataFrame, pd.Series], actual: Union[pd.DataFrame, pd.Series], target_col: str, buckets: int = 10, equal_freq: bool = True, group_by: Optional[Union[str, List[str]]] = None, return_details: bool = False, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5) -> Union[float, pd.DataFrame, Tuple[Dict, Dict]]: ...
def calculate_within_psi(data: pd.DataFrame, grp_name: str, target_col: str, benchmark: Optional[Any] = None, equal_freq: bool = True, buckets: int = 10, return_details: bool = False, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5, benchmark_display_name: Optional[str] = None) -> Union[pd.DataFrame, Dict]: ...
def calculate_psi_within_dataset(data: pd.DataFrame, grp_name: str, varlist: List[str], benchmark: Optional[Any] = None, equal_freq: bool = True, buckets: int = 10, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5) -> pd.DataFrame: ...
def calculate_multivar_psi_two_sets(expected_df: pd.DataFrame, actual_df: pd.DataFrame, varlist: List[str], group_by: Optional[Union[str, List[str]]] = None, buckets: int = 10, equal_freq: bool = True, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5) -> pd.DataFrame: ...
def calculate_multigroup_psi_two_sets(expected_df: pd.DataFrame, actual_df: pd.DataFrame, varlist: List[str], group_by: Optional[Union[str, List[str]]] = None, buckets: int = 10, equal_freq: bool = True, min_bin_prop: float = 0.05, content: float = 1e-06, precision: int = 5, group_name: Optional[str] = None, return_details: bool = False) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]: ...
