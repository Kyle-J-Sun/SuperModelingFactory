from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ._common import make_dirs, safe_to_csv


@dataclass
class ScoreConsistencyUATPipelineConfig:
    output_dir: str = "output/score_consistency_uat"
    random_state: int = 42
    write_outputs: bool = True
    write_excel: bool = True

    sql_dir: str = "sql"
    offline_sql: str = "pull_offline.sql"
    online_sql: str = "pull_online.sql"
    sqlrunner: Any | None = None
    env_path: str | None = None
    n_process: int | str | None = "auto"
    offline_data: pd.DataFrame | None = None
    online_data: pd.DataFrame | None = None

    main_model_score_col: str = "credit_risk_v31_cdc_submodel_score"
    tol_score: float = 1e-6
    tol_feat: float = 1e-2
    time_featlist: list[str] = field(default_factory=list)
    tol_time_seconds: float = 60.0

    excel_output_path: str | None = None
    excel_font: str = "Arial"
    info_list: list[str] = field(default_factory=list)

    include_submodel_scores: bool = True
    submodel_pairs: dict[str, str] = field(default_factory=dict)


@dataclass
class ScoreConsistencyUATPipelineResult:
    offline_data: pd.DataFrame | None
    online_data: pd.DataFrame | None
    compare_data: pd.DataFrame | None
    both_data: pd.DataFrame | None
    coverage_summary: dict[str, Any]
    main_score_summary: dict[str, Any]
    submodel_summary: list[dict[str, Any]]
    feature_diff_summary: pd.DataFrame
    time_summary: pd.DataFrame
    per_flow_report: pd.DataFrame
    summary: pd.DataFrame
    report_path: str | None = None
    checker: Any | None = None


class ScoreConsistencyUATPipeline:
    """Reusable online/offline score consistency UAT workflow."""

    def __init__(self, config: ScoreConsistencyUATPipelineConfig | None = None):
        self.config = config or ScoreConsistencyUATPipelineConfig()

    def run(
        self,
        offline_data: pd.DataFrame | None = None,
        online_data: pd.DataFrame | None = None,
    ) -> ScoreConsistencyUATPipelineResult:
        from Modeling_Tool.UAT import UATConsistencyChecker

        cfg = self.config
        self._load_env(cfg.env_path)
        report_dir = Path(cfg.output_dir) / "report"
        if cfg.write_outputs or cfg.write_excel:
            make_dirs(cfg.output_dir, report_dir)

        offline = offline_data if offline_data is not None else cfg.offline_data
        online = online_data if online_data is not None else cfg.online_data
        use_dataframes = offline is not None or online is not None
        checker = UATConsistencyChecker(self._build_uat_config(), self._resolve_sqlrunner(use_dataframes))

        if use_dataframes:
            if offline is None or online is None:
                raise ValueError("Provide both offline_data and online_data for DataFrame mode.")
            self._load_dataframes(checker, offline, online)
        else:
            checker.load_data()

        coverage_summary = checker.check_coverage()
        main_score_summary = checker.check_main_score()
        submodel_summary = checker.check_submodel_features()
        feature_diff_summary = checker.check_all_features()
        time_summary = checker.check_time_fields()
        per_flow_report = checker.build_per_flow_report()
        summary = checker.build_summary()

        report_path = None
        if cfg.write_outputs:
            safe_to_csv(pd.DataFrame([coverage_summary]), report_dir / "coverage_summary.csv", index=False)
            safe_to_csv(pd.DataFrame([main_score_summary]), report_dir / "main_score_summary.csv", index=False)
            safe_to_csv(pd.DataFrame(submodel_summary), report_dir / "submodel_summary.csv", index=False)
            safe_to_csv(feature_diff_summary, report_dir / "feature_diff_summary.csv", index=False)
            safe_to_csv(time_summary, report_dir / "time_summary.csv", index=False)
            safe_to_csv(per_flow_report, report_dir / "per_flow_report.csv", index=False)
            safe_to_csv(summary, report_dir / "summary.csv", index=False)

        if cfg.write_excel:
            report_path = checker.export_excel()

        return ScoreConsistencyUATPipelineResult(
            offline_data=checker.df_offline,
            online_data=checker.df_online,
            compare_data=checker.df_compare,
            both_data=checker.df_both,
            coverage_summary=coverage_summary,
            main_score_summary=main_score_summary,
            submodel_summary=submodel_summary,
            feature_diff_summary=feature_diff_summary,
            time_summary=time_summary,
            per_flow_report=per_flow_report,
            summary=summary,
            report_path=report_path,
            checker=checker,
        )

    def _build_uat_config(self):
        from Modeling_Tool.UAT import UATConfig

        cfg = self.config
        return UATConfig(
            main_model_score_col=cfg.main_model_score_col,
            include_submodel_scores=cfg.include_submodel_scores,
            excel_output_path=self._resolve_excel_output_path(),
            sql_dir=str(Path(cfg.sql_dir).expanduser().resolve()),
            offline_sql=cfg.offline_sql,
            online_sql=cfg.online_sql,
            tol_score=float(cfg.tol_score),
            tol_feat=float(cfg.tol_feat),
            n_process=self._resolve_n_process(cfg.n_process),
            submodel_pairs=dict(cfg.submodel_pairs or {}),
            excel_font=cfg.excel_font,
            info_list=list(cfg.info_list or []),
            time_featlist=list(cfg.time_featlist or []),
            tol_time_seconds=float(cfg.tol_time_seconds),
        )

    def _resolve_excel_output_path(self) -> str:
        cfg = self.config
        if cfg.excel_output_path:
            path = Path(os.path.expandvars(cfg.excel_output_path)).expanduser()
        else:
            path = Path(cfg.output_dir) / "report" / "Score_Consistency_UAT_Report.xlsx"
        if not path.is_absolute():
            path = path.resolve()
        if cfg.write_excel:
            path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def _resolve_n_process(self, value: int | str | None) -> int:
        if value in (None, "auto"):
            try:
                import multiprocessing

                return max(1, multiprocessing.cpu_count() - 1)
            except Exception:
                return 1
        if isinstance(value, int) and value >= 1:
            return value
        raise ValueError("n_process must be a positive integer, null, or 'auto'.")

    def _resolve_sqlrunner(self, use_dataframes: bool = False) -> Any:
        cfg = self.config
        if cfg.sqlrunner is not None:
            return cfg.sqlrunner
        if use_dataframes:
            return _NoOpSQLRunner()
        from Modeling_Tool.Core import ODPSRunner

        return ODPSRunner()

    def _load_env(self, env_path: str | None) -> None:
        if not env_path:
            return
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ImportError("python-dotenv is required when env_path is provided.") from exc
        path = Path(os.path.expandvars(env_path)).expanduser()
        load_dotenv(path, override=False)

    def _load_dataframes(
        self,
        checker: Any,
        offline_data: pd.DataFrame,
        online_data: pd.DataFrame,
    ) -> None:
        for name, frame in {"offline_data": offline_data, "online_data": online_data}.items():
            if "flow_id" not in frame.columns:
                raise KeyError(f"{name} must contain a flow_id column.")

        checker.df_offline = offline_data.copy()
        checker.df_online = online_data.copy()
        online_extra = [col for col in checker.df_online.columns if col != "flow_id"]
        checker.df_compare = checker.df_offline.merge(
            checker.df_online[["flow_id"] + online_extra],
            on="flow_id",
            how="outer",
            suffixes=("", "_online"),
            indicator=True,
        )

        for col in checker.df_compare.columns:
            if col in ("flow_id", "_merge"):
                continue
            if checker.df_compare[col].dtype == object:
                as_num = pd.to_numeric(checker.df_compare[col], errors="coerce")
                if as_num.notna().any():
                    checker.df_compare[col] = as_num

        checker.df_both = checker.df_compare[checker.df_compare["_merge"] == "both"].copy()
        checker._info_cols = [
            col for col in checker.cfg.info_list if col != "flow_id" and col in checker.df_both.columns
        ]


class _NoOpSQLRunner:
    def run_sql(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        raise RuntimeError("sqlrunner is required when offline_data/online_data are not provided.")
