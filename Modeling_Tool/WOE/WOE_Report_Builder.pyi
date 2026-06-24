# =============================================================================
# Modeling_Tool.WOE.WOE_Report_Builder
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-WOEREPORTBUI-85e09901
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import os
import logging
import pandas as pd
from ExcelMaster.ExcelMaster import ExcelMaster
def get_woe_plot_report_new(em, ws, woe_plot_dir, grp_name, varlist, means_rpt = None, var_dict = None): ...

class WoeReportBuilder:
    def __init__(self, em, data, valid_varlist: list, woe_suffix: str = '_woe', proc_means_func = None, missing_rate_ref = 0.95, default_var_dict: dict = None): ...
    def add_group(self, group_name: str, woe_plot_dir: str, sheet_name: str = None, cell_scale: tuple = (1, 2), var_dict: dict = None): ...
    def close(self): ...
