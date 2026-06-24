import optuna
import os,sys
import random
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# from model_tool import *
# import util_funcs as uf
from utils import mkdir_if_not_exist

sys.path.insert(0, "/opt/workspace/sunjingkai_personal_drive/git-repo/")
from cdaml.modelutils.evaluate import evaluate_model
from cdaml.modelutils.models import lgb_utils

# sys.path.insert(0, "/opt/workspace/sunjingkai_personal_drive/git-repo/cdaml_demo/")
# from cdaml.modelutils.var_analysis.VarStats import VarStats
# import cdaml.modelutils.var_analysis.woe_tool as woe

# def woe_binning(df, dep, varlist, grp_name, dtt_name, result_dir, 
#                 missing_rate_ref = -99999, oot_start_dt = None, digit = 5, 
#                 replace_dir = False, init_bins = 100, min_prop_per_bin = 0.05, 
#                 chi2_p = 0.95, max_bins = 10, spec_values = None, plot_min_iv = 0.005):
#     """ WOE Binning using Chi-Squared Method. """
    
#     mkdir_if_not_exist(result_dir, replace = replace_dir)
    
#     sample_cols = [x for x in df.columns if x not in varlist]
    
#     if oot_start_dt is not None:
#         # 需要在modeling/config.py文件中设置这些值
#         var_stats = VarStats(
#             master_df=df, 
#             ignore_names=sample_cols, #不跑iv的字段
#             grp_name=grp_name, # 分组的字段名
#             dt_name=dtt_name, # 样本时间
#             oot_start_dt=oot_start_dt, # oot开始时间
#             digit=digit, # 保留小数位数
#             analysis_dir=result_dir #结果保存地址
#         )
        
#     else:
#         # 需要在modeling/config.py文件中设置这些值
#         var_stats = VarStats(
#             master_df=df, 
#             ignore_names=sample_cols, #不跑iv的字段
#             grp_name=grp_name, # 分组的字段名
#             dt_name=dtt_name, # 样本时间
#             train_dt_list=[df[dtt_name].min(), df[dtt_name].max()], ## 全量样本
#             digit=digit, # 保留小数位数
#             analysis_dir=result_dir #结果保存地址
#         )

#     # !! Issue with mean, min values: Include specified missing value into the calculation !!
#     var_stats.calc_numvars_stats(
#         num_metric_list=["missrate", "unique", "range", "quantile"], 
#         num_missing_values=[missing_rate_ref], 
#         cn_header=False
#     )

#     # 计算变量之间的相关性
#     var_stats.calc_numvars_corr()

#     # 计算变量iv值，画WOE
#     var_stats.calc_numvars_iv(
#         tgt_name=dep, 
#         spec_values=spec_values, # 特殊值单独分箱
#         max_bins=max_bins,  # 最大分箱值
#         equi_method="equif",  #等频分组
#         min_prop_in_bin=min_prop_per_bin, # 每个分箱中样本量最少占比
#         equi_bins=init_bins, 
#         binning_criteria="chi2", 
#         chi2_p=chi2_p, 
#         plot_min_iv=plot_min_iv
#     )
    
#     return None