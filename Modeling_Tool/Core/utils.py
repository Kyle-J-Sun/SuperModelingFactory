import subprocess
import os, sys, logging
logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
from pandas import *

from datetime import date as dt
from dateutil.relativedelta import relativedelta as rd

from .kDataFrame import kDataFrame

from .sample_weight_utils import (
    resolve_sample_weight,
    validate_sample_weight,
    weighted_sum,
    weighted_mean,
    weighted_rate,
)

def bucket_by_cond(df: pd.DataFrame, cond_dict: dict, colname: str, 
                   drop_unmatched: bool = True, default=np.nan) -> pd.DataFrame:
    """
    根据 query 条件字典对数据分组打标签。
    
    Parameters
    ----------
    df : 原始 DataFrame
    cond_dict : {标签: query条件字符串}
    colname : 新增的标签列名
    drop_unmatched : True=丢弃未命中行(原逻辑), False=保留全量
    default : 未命中行的默认值（仅 drop_unmatched=False 时生效）
    """
    if drop_unmatched:
        res_list = []
        for label, cond in cond_dict.items():
            sub = df.query(cond).copy()
            sub[colname] = label
            res_list.append(sub)
        return pd.concat(res_list)
    else:
        result = df.copy()
        result[colname] = default
        for label, cond in cond_dict.items():
            mask = result.eval(cond)
            result.loc[mask, colname] = label
        return result