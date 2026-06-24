# =============================================================================
# Modeling_Tool.Eval.Model_Eval_Tool
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-MODELEVALTOO-09e04fef
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import logging
import os
import numpy as np
import pandas as pd
from Modeling_Tool.Core.Binning_Tool import get_bin_range_list, super_binning
from Modeling_Tool.Core.utils import load_model, calc_iv, calc_woe
from .evaluate_model import evaluate_performance
def get_gains_table(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, score = None, model = None, varlist = None, equal_freq = True, chi2_method = False, grp_name = None, min_data_size = 100, grp_colname = None, sync_range = True, chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], retSummary = False, tree_binning = False, random_state = 42, ascending = False, withSummary = False, wholeGroup = False, add_func = None): ...
def get_perf_summary(train, validation, oot, tgt_name, scr_name = None, model = None, feature_cols = None, fig_save_path = None, rpt_save_path = None, to_show = False, display = True, dist_bins = 20, pct_bins = 10, precision = 5, min_bin_prop = 0.05, include_missing = False, equal_freq = True, chi2_method = False, init_equi_bins = 1000, chi2_p = 0.9, oot_grp_name = None, min_data_size = 100, grp_colname = None, tree_binning = False, random_state = 42, gains_table = False): ...
def cross_risk(data, score_list, dep, nbins, agg_col = None, precision = 5, min_bin_prop = 0.05, include_missing = False, equal_freq = True, binning_numeric = [True, True], agg_func = 'mean', chi2_method = False, chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], tree_binning = False, random_state = 42): ...
def get_gains_table_by_cust_metrics(data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, score = None, model = None, varlist = None, equal_freq = True, chi2_method = False, grp_name = None, min_data_size = 100, grp_colname = None, sync_range = True, chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], tree_binning = False, random_state = 42, ascending = True, eval_metrics = ['age', 'monthly_income', 'education'], metric_agg_func = 'mean', withSummary = False): ...
def tie_score_rate(data, score): ...
def score_unique_rate(data, score): ...

class GainsTableCalculator:
    def __init__(self, data, dep, nbins = 10, precision = 5, min_bin_prop = 0.05, include_missing = True, score = None, model = None, varlist = None, equal_freq = True, chi2_method = False, chi2_p = 0.95, init_equi_bins = 100, fillna = -999999, spec_values = [], tree_binning = False, random_state = 42, ascending = False): ...
    def calculate(self, grp_name = None, min_data_size = 100, grp_colname = None, sync_range = True, retSummary = False, withSummary = False, wholeGroup = False, add_func = None): ...

class PerformanceEvaluator:
    def __init__(self, tgt_name, scr_name = None, model = None, feature_cols = None, dist_bins = 20, pct_bins = 10, precision = 5, min_bin_prop = 0.05, include_missing = False, equal_freq = True, chi2_method = False, init_equi_bins = 1000, chi2_p = 0.9, tree_binning = False, random_state = 42): ...
    def add_dataset(self, name, data): ...
    def evaluate(self, oot_grp_name = None, min_data_size = 100, grp_colname = None, fig_save_path = None, rpt_save_path = None, to_show = False, display = True, gains_table = False, benchmark_dataset = None): ...
