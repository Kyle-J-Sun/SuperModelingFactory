# =============================================================================
# Modeling_Tool.Feature.Feature_Insights
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-FEATUREINSIG-9fafdb0f
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import pandas as pd
from tqdm import tqdm
from .Distribution_Tool import proc_means_by_grp
import logging

class VarExtractionInsights:
    def __init__(self, data, dep, plot_path, nbins = 10, equal_freq = True, min_bin_prop = 0.05, precision = 5, chi2_method = False, chi2_p = 0.9, init_equi_bins = 5000, tree_binning = True, include_missing = True, seed = 3407, missing_rate_ref = -999999, spec_values = None): ...
    def remove_folder(file_path): ...
    def get_var_analysis_report(self, data, varlist, dep = None, iv_cut = 0.01): ...
    def plot_woe(self, data, varlist, plot_group = None, plot_dirname = 'var_analysis_plot', plot_path = None): ...
def var_corr_filter(data, varlist, corr_cutpoint = 0.8, method = 'pearson'): ...

class CorrelationFilter:
    def __init__(self, data, dep, corr_cutpoint = 0.8, method = 'pearson', tree_binning = False, chi2_method = False, seed = 42, chi2_p = 0.999, init_equi_bins = 1000, missing_rate_ref = -9999999, spec_values = [], base_metric = 'iv'): ...
    def filter_single_iteration(self, varlist): ...
    def remove_highly_correlated(self, varlist, max_iterations = 10): ...
    def calculate_vif(df): ...
