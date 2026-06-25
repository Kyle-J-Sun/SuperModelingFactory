# =============================================================================
# Modeling_Tool.WOE.plot_woe_tool
# -----------------------------------------------------------------------------
# Copyright (c) 2026 Kyle Sun <github.com/Kyle-J-Sun>. All rights reserved.
# SuperModelingFactory — Licensed under the Business Source License 1.1.
#
# This stub describes the public API of a closed-source module compiled to a
# native extension (.so / .pyd). The original source is not distributed.
# Production / commercial use requires a separate commercial license.
#
# FINGERPRINT: SMF-PLOTWOETOOL-98e549ad
#   (Unique trace marker. Do not remove or alter — used for plagiarism
#    detection across the public internet.)
# =============================================================================

import os
import pandas as pd
import numpy as np
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from Modeling_Tool.Core.utils import calc_iv
def extract_group_value(woe_grp_df, value_name = 'lift'): ...
def cre_psi_table(woe_grp_df, exp_values, value_name = 'p'): ...
def plot_woe(woe_df, var_rename = None, to_show = True, save_dir = None): ...
def plot_woe_group(woe_grp_df, var_rename = None, to_show = True, save_dir = None): ...
