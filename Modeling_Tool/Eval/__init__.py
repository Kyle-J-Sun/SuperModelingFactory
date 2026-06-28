from .Model_Eval_Tool import (
    get_gains_table,
    get_perf_summary,
    cross_risk,
    tie_score_rate,
    score_unique_rate,
    get_gains_table_by_cust_metrics,
#     get_backward_perf,
#     get_backward_summary,
    GainsTableCalculator,
    PerformanceEvaluator,
#     BackwardEliminationAnalyzer
)

from .Evaluation_Tool import (
    EvaluationPipeline,
    Model_Evaluation_Tool,
    Utility_Functions,
)

from .evaluate_model import (
    calc_pr,
    summarize_pr,
    plot_pr_curve,
    calc_roc,
    plot_ks_curve,
    plot_roc_curve,
    calc_equid_dist,
    plot_kde_curve,
    plot_dist_curve,
    calc_equid_pct,
    summarize_pct,
    plot_pct_curve,
    calc_fixed_pct,
    plot_cumdist_curve,
    plot_cumpct_curve,
    plot_gain_curve,
    evaluate_performance,
    evaluate_distribution,
    comparison_performance,
    calc_lift_apt,
    resturct_gains,
)

from Modeling_Tool.weighted_integration import apply_eval_patches
apply_eval_patches(globals())

__all__ = [
    # Model_Eval_Tool
    'get_gains_table', 'get_perf_summary', 'cross_risk', 'tie_score_rate',
    'score_unique_rate', 'get_gains_table_by_cust_metrics',
#     'get_backward_perf', 'get_backward_summary',
    'GainsTableCalculator', 'PerformanceEvaluator',
#     'BackwardEliminationAnalyzer',

    # Evaluation_Tool
    'EvaluationPipeline', 'Model_Evaluation_Tool', 'Utility_Functions',

    # evaluate_model
    'calc_pr', 'summarize_pr', 'plot_pr_curve',
    'calc_roc', 'plot_ks_curve', 'plot_roc_curve',
    'calc_equid_dist', 'plot_kde_curve', 'plot_dist_curve',
    'calc_equid_pct', 'summarize_pct', 'plot_pct_curve',
    'calc_fixed_pct', 'plot_cumdist_curve', 'plot_cumpct_curve', 'plot_gain_curve',
    'evaluate_performance', 'evaluate_distribution', 'comparison_performance',
    'calc_lift_apt', 'resturct_gains',
]
