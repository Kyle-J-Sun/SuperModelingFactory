# =============================================================================
# Modeling_Tool.Sample.Reject_Infer
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-REJECTINFER-31339160
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

class RejectInferrer(ABC):
    def __init__(self, target_col: str = 'target', score_col: str = 'score'): ...
    def infer(self, df_approved: pd.DataFrame, df_rejected: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame: ...

class SimpleAugmentInferrer(RejectInferrer):
    def __init__(self, target_col: str = 'target', score_col: str = 'score', bad_rate: Optional[float] = None): ...
    def infer(self, df_approved: pd.DataFrame, df_rejected: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame: ...

class HardCutoffInferrer(RejectInferrer):
    def __init__(self, target_col: str = 'target', score_col: str = 'score', cutoff: float = 0.5): ...
    def infer(self, df_approved: pd.DataFrame, df_rejected: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame: ...

class FuzzyAugmentInferrer(RejectInferrer):
    def __init__(self, target_col: str = 'target', score_col: str = 'score', weight_factor: float = 1.0): ...
    def infer(self, df_approved: pd.DataFrame, df_rejected: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame: ...

class ParcelingInferrer(RejectInferrer):
    def __init__(self, target_col: str = 'target', score_col: str = 'score', n_parcels: int = 10): ...
    def infer(self, df_approved: pd.DataFrame, df_rejected: pd.DataFrame, score_col: Optional[str] = None) -> pd.DataFrame: ...

class RejectInferenceFactory:
    def create(cls, method: str = 'parceling', **kwargs) -> RejectInferrer: ...
    def available_methods(cls) -> List[str]: ...
