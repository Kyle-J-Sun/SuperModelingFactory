from .Binning_Tool import (
    Binning,
    chi2_binning,
    run_binning,
    super_binning,
    get_max_nbins,
    get_decision_tree_binning_edges,
    NumVarBinning,
    cre_pvt,
    merge_bins,
    observed_laplace,
    cat_2_list,
    get_bin_range,
    get_bin_range_list,
    chi2_auto_binning,
    quick_binning,
)

def __getattr__(name):
    if name == "ODPSRunner":
        from .ODPS_Tool import ODPSRunner as _ODPSRunner
        return _ODPSRunner
    if name in {"ParallelODPSConfig", "ParallelODPSManager", "ParallelODPSPuller"}:
        from . import Parallel_ODPS_Manager as _parallel_odps
        return getattr(_parallel_odps, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

from .Slope_Tool import (
    calculate_slope_sklearn,
    calculate_slope_scipy,
    calculate_slope_numpy,
    calculate_slope_manual,
    SlopeCalculator,
)

from .sample_weight_utils import (
    resolve_sample_weight,
    validate_sample_weight,
    weighted_sum,
    weighted_mean,
    weighted_rate,
)

from .Parallel_Engine import (
    ParallelApplyConfig,
    ParallelApplyEngine,
    ParallelApplyResult,
    parallel_apply,
)

from .utils import (
    cut2pieces,
    proc_freq,
    read_attr_list,
    write_attr_list,
    odds_score,
    parse_odps_schema,
    npnan2none,
    drop_tmp_cols,
    mkdir_if_not_exist,
    parse_sql_file,
    calc_woe,
    calc_iv,
    scoring,
    get_missing_indicator,
    upload_score,
    DataFrameProcessor,
    FilePathManager,
    DateTimeUtils,
    WOEIVCalculator,
    get_feature_names,
    get_feature_names_lgb,
    get_feature_names_xgb,
    get_feature_names_batch,
    pull_attributes_in_batch,
)
from .Model_Registry_Tool import (
    make_model_artifact,
    save_model,
    load_model,
    load_model_metadata,
)

from .XOR_Encryptor import TextEncryptor

from .kDataFrame import kSeries, kDataFrame

__all__ = [
    'Binning', 'chi2_binning', 'run_binning', 'super_binning',
    'get_max_nbins', 'get_decision_tree_binning_edges', 'NumVarBinning',
    'cre_pvt', 'merge_bins', 'observed_laplace', 'cat_2_list',
    'get_bin_range', 'get_bin_range_list', 'chi2_auto_binning', 'quick_binning',
    'ODPSRunner',
    'calculate_slope_sklearn', 'calculate_slope_scipy',
    'calculate_slope_numpy', 'calculate_slope_manual',
    'SlopeCalculator',
    'resolve_sample_weight', 'validate_sample_weight',
    'weighted_sum', 'weighted_mean', 'weighted_rate',
    'ParallelApplyConfig', 'ParallelApplyEngine', 'ParallelApplyResult',
    'parallel_apply',
    'ParallelODPSConfig', 'ParallelODPSManager',
    'cut2pieces', 'proc_freq', 'read_attr_list', 'write_attr_list',
    'odds_score', 'parse_odps_schema', 'npnan2none', 'drop_tmp_cols',
    'mkdir_if_not_exist', 'parse_sql_file', 'calc_woe', 'calc_iv',
    'save_model', 'load_model', 'load_model_metadata', 'make_model_artifact',
    'scoring', 'get_missing_indicator', 'upload_score',
    'get_feature_names', 'get_feature_names_lgb', 'get_feature_names_xgb',
    'get_feature_names_batch', 'pull_attributes_in_batch',
    'DataFrameProcessor', 'FilePathManager', 'DateTimeUtils', 'WOEIVCalculator',
    'TextEncryptor',
    'kSeries', 'kDataFrame',
]
