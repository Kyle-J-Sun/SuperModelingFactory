from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SampleAnalysisPipelineConfig:
    target_cols: list[str] = field(
        default_factory=lambda: [
            "y_flag_dpd7_in_mob1",
            "y_flag_dpd7_in_mob3",
            "y_flag_dpd7_in_mob6",
            "y_flag_dpd7_in_mob12",
        ]
    )
    time_col: str = "apply_time"
    time_dims: list[str] = field(default_factory=lambda: ["apply_week", "apply_month", "apply_quarter"])
    population_dims: list[str] = field(default_factory=lambda: ["channel", "strategy_version"])
    profile_cols: list[str] = field(default_factory=lambda: ["age", "income", "education", "credit_limit"])
    oot_time_dim: str = "apply_month"
    oot_windows: list[int] = field(default_factory=lambda: [1, 2, 3, 6])
    ins_oos_ratios: list[float] = field(default_factory=lambda: [0.7, 0.75, 0.8])
    random_seeds: range | list[int] | tuple[int, ...] = field(default_factory=lambda: range(3000, 3020))
    min_sample_size: int = 500
    output_dir: str = "output/sample_analysis"
    write_outputs: bool = True
    write_excel: bool = True


@dataclass
class SampleAnalysisPipelineResult:
    label_coverage_summary: pd.DataFrame
    segment_bad_rate_summary: pd.DataFrame
    profile_summary: pd.DataFrame
    split_candidate_summary: pd.DataFrame
    split_recommendation: pd.DataFrame
    output_paths: dict[str, str]


class SampleAnalysisPipeline:
    """Analyze label maturity, segment drift, and INS/OOS/OOT split stability."""

    def __init__(self, config: SampleAnalysisPipelineConfig | None = None):
        self.config = config or SampleAnalysisPipelineConfig()

    def run(self, data: pd.DataFrame) -> SampleAnalysisPipelineResult:
        self._validate_input(data)
        work = data.copy()
        work[self.config.time_col] = pd.to_datetime(work[self.config.time_col])

        label_coverage = self._label_coverage(work)
        segment_bad_rate = self._segment_bad_rate(work)
        profile_summary = self._profile_summary(work)
        split_candidates = self._split_candidates(work)
        split_recommendation = self._recommend_splits(split_candidates)
        output_paths = self._write_outputs(
            {
                "label_coverage_summary": label_coverage,
                "segment_bad_rate_summary": segment_bad_rate,
                "profile_summary": profile_summary,
                "split_candidate_summary": split_candidates,
                "split_recommendation": split_recommendation,
            }
        )

        return SampleAnalysisPipelineResult(
            label_coverage_summary=label_coverage,
            segment_bad_rate_summary=segment_bad_rate,
            profile_summary=profile_summary,
            split_candidate_summary=split_candidates,
            split_recommendation=split_recommendation,
            output_paths=output_paths,
        )

    def _validate_input(self, data: pd.DataFrame) -> None:
        cfg = self.config
        required = (
            [cfg.time_col, cfg.oot_time_dim]
            + cfg.target_cols
            + cfg.time_dims
            + cfg.population_dims
            + cfg.profile_cols
        )
        missing = [col for col in dict.fromkeys(required) if col not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

    def _label_coverage(self, data: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        rows = []
        for target in cfg.target_cols:
            mature = data[data[target].notna()]
            rows.append(
                {
                    "target_col": target,
                    "n_total": len(data),
                    "n_observed": len(mature),
                    "observed_rate": len(mature) / len(data) if len(data) else np.nan,
                    "bad_rate": mature[target].mean(),
                    "apply_time_min": mature[cfg.time_col].min(),
                    "apply_time_max": mature[cfg.time_col].max(),
                    "n_approved_observed": int(mature["is_approved"].sum()) if "is_approved" in mature else np.nan,
                }
            )
        return pd.DataFrame(rows)

    def _segment_bad_rate(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for target in self.config.target_cols:
            mature = data[data[target].notna()].copy()
            rows.extend(self._summarize_bad_rate(mature, target, [], "global"))
            for dim in self.config.time_dims:
                rows.extend(self._summarize_bad_rate(mature, target, [dim], "time"))
            for dim in self.config.population_dims:
                rows.extend(self._summarize_bad_rate(mature, target, [dim], "population"))
            for time_dim in self.config.time_dims:
                for pop_dim in self.config.population_dims:
                    rows.extend(self._summarize_bad_rate(mature, target, [time_dim, pop_dim], "time_x_population"))
        return pd.DataFrame(rows)

    def _profile_summary(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = []
        group_specs = [([], "global")]
        group_specs.extend(([dim], "time") for dim in self.config.time_dims)
        group_specs.extend(([dim], "population") for dim in self.config.population_dims)
        group_specs.extend(
            ([time_dim, pop_dim], "time_x_population")
            for time_dim in self.config.time_dims
            for pop_dim in self.config.population_dims
        )

        for target in self.config.target_cols:
            mature = data[data[target].notna()].copy()
            for group_cols, group_type in group_specs:
                rows.extend(self._summarize_profile(mature, target, list(group_cols), group_type))
        return pd.DataFrame(rows)

    def _split_candidates(self, data: pd.DataFrame) -> pd.DataFrame:
        from Modeling_Tool import SampleSplitter

        cfg = self.config
        rows = []
        for target in cfg.target_cols:
            mature = data[data[target].notna()].copy()
            time_values = sorted(mature[cfg.oot_time_dim].dropna().unique())
            if not time_values:
                continue
            for oot_window in cfg.oot_windows:
                oot_values = set(time_values[-int(oot_window):])
                oot = mature[mature[cfg.oot_time_dim].isin(oot_values)].copy()
                ins_oos_pool = mature[~mature[cfg.oot_time_dim].isin(oot_values)].copy()
                if len(oot) == 0 or len(ins_oos_pool) == 0:
                    continue

                for ins_ratio in cfg.ins_oos_ratios:
                    for seed in list(cfg.random_seeds):
                        splitter = SampleSplitter(
                            test_size=1 - float(ins_ratio),
                            random_state=int(seed),
                            stratify=True,
                        )
                        try:
                            ins, oos = splitter.split_df(ins_oos_pool, target=target)
                        except ValueError:
                            splitter = SampleSplitter(
                                test_size=1 - float(ins_ratio),
                                random_state=int(seed),
                                stratify=False,
                            )
                            ins, oos = splitter.split_df(ins_oos_pool, target=target)

                        br_ins = ins[target].mean()
                        br_oos = oos[target].mean()
                        br_oot = oot[target].mean()
                        rows.append(
                            {
                                "target_col": target,
                                "oot_time_dim": cfg.oot_time_dim,
                                "oot_window_periods": int(oot_window),
                                "oot_window_months": int(oot_window),
                                "oot_periods": ",".join(str(x) for x in sorted(oot_values)),
                                "oot_months": ",".join(str(x) for x in sorted(oot_values)),
                                "ins_ratio": float(ins_ratio),
                                "oos_ratio": 1 - float(ins_ratio),
                                "seed": int(seed),
                                "n_ins": len(ins),
                                "n_oos": len(oos),
                                "n_oot": len(oot),
                                "bad_rate_ins": br_ins,
                                "bad_rate_oos": br_oos,
                                "bad_rate_oot": br_oot,
                                "abs_diff_ins_oos": abs(br_ins - br_oos),
                                "abs_diff_oos_oot": abs(br_oos - br_oot),
                                "max_abs_bad_rate_gap": max(
                                    abs(br_ins - br_oos),
                                    abs(br_ins - br_oot),
                                    abs(br_oos - br_oot),
                                ),
                                "oot_sample_pct": len(oot) / len(mature) if len(mature) else np.nan,
                            }
                        )
        return pd.DataFrame(rows)

    def _recommend_splits(self, split_candidates: pd.DataFrame) -> pd.DataFrame:
        if split_candidates.empty:
            return split_candidates.copy()

        cfg = self.config
        recs = []
        for _, sub in split_candidates.groupby("target_col", dropna=False):
            valid = sub[
                (sub["n_ins"] >= cfg.min_sample_size)
                & (sub["n_oos"] >= cfg.min_sample_size)
                & (sub["n_oot"] >= cfg.min_sample_size)
            ].copy()
            if valid.empty:
                valid = sub.copy()
            valid["ratio_distance_to_75_25"] = (valid["ins_ratio"] - 0.75).abs()
            valid = valid.sort_values(
                ["max_abs_bad_rate_gap", "n_oot", "ratio_distance_to_75_25"],
                ascending=[True, False, True],
            )
            rec = valid.head(1).copy()
            rec["recommend_reason"] = (
                "Min bad-rate gap after sample-size filter; tie-break by larger OOT and 75/25 proximity"
            )
            recs.append(rec)
        return pd.concat(recs, ignore_index=True)

    def _summarize_bad_rate(
        self,
        data: pd.DataFrame,
        target: str,
        group_cols: list[str],
        group_type: str,
    ) -> list[dict[str, Any]]:
        if not group_cols:
            return [
                {
                    "target_col": target,
                    "group_type": group_type,
                    "group_cols": "global",
                    "group_value": "ALL",
                    "n": len(data),
                    "bad_rate": data[target].mean(),
                }
            ]

        try:
            from Modeling_Tool import EvaluationPipeline

            class _EvalDataHolder:
                def __init__(self, frame: pd.DataFrame):
                    self.data = frame

            def _bad_rate_eval(current_data: pd.DataFrame) -> pd.DataFrame:
                return pd.DataFrame(
                    {
                        "n": [len(current_data)],
                        "bad_rate": [current_data[target].mean()],
                    }
                )

            pipeline = EvaluationPipeline(_EvalDataHolder(data))
            for col in group_cols:
                pipeline = pipeline.group_by(col, min_size=1, group_var_name=col)
            grouped_summary = pipeline.apply(_bad_rate_eval)
            if isinstance(grouped_summary, pd.DataFrame) and not grouped_summary.empty:
                rows = []
                for _, row in grouped_summary.iterrows():
                    rows.append(
                        {
                            "target_col": target,
                            "group_type": group_type,
                            "group_cols": " x ".join(group_cols),
                            "group_value": " x ".join(str(row[col]) for col in group_cols),
                            "n": int(row["n"]),
                            "bad_rate": float(row["bad_rate"]),
                        }
                    )
                return rows
        except Exception:
            pass

        rows = []
        for key, sub in data.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            rows.append(
                {
                    "target_col": target,
                    "group_type": group_type,
                    "group_cols": " x ".join(group_cols),
                    "group_value": " x ".join(str(x) for x in key),
                    "n": len(sub),
                    "bad_rate": sub[target].mean(),
                }
            )
        return rows

    def _summarize_profile(
        self,
        data: pd.DataFrame,
        target: str,
        group_cols: list[str],
        group_type: str,
    ) -> list[dict[str, Any]]:
        rows = []
        if not group_cols:
            groups = [(("ALL",), data)]
            group_cols_label = "global"
        else:
            groups = data.groupby(group_cols, dropna=False)
            group_cols_label = " x ".join(group_cols)

        for key, sub in groups:
            if not isinstance(key, tuple):
                key = (key,)
            row = {
                "target_col": target,
                "group_type": group_type,
                "group_cols": group_cols_label,
                "group_value": " x ".join(str(x) for x in key),
                "n": len(sub),
                "bad_rate": sub[target].mean(),
            }
            for col in self.config.profile_cols:
                row[f"{col}_mean"] = sub[col].mean()
                row[f"{col}_median"] = sub[col].median()
            rows.append(row)
        return rows

    def _write_outputs(self, tables: dict[str, pd.DataFrame]) -> dict[str, str]:
        cfg = self.config
        output_dir = Path(cfg.output_dir)
        output_paths: dict[str, str] = {}
        if not (cfg.write_outputs or cfg.write_excel):
            return output_paths

        output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.write_outputs:
            for name, df in tables.items():
                path = output_dir / f"{name}.csv"
                df.to_csv(path, index=False)
                output_paths[name] = str(path.resolve())

        if cfg.write_excel:
            excel_path = output_dir / "Sample_Analysis_Report.xlsx"
            self._write_excel_report(excel_path, tables)
            output_paths["excel_report"] = str(excel_path.resolve())
        return output_paths

    def _write_excel_report(self, excel_path: Path, tables: dict[str, pd.DataFrame]) -> None:
        from ExcelMaster.ExcelMaster import ExcelMaster

        em = ExcelMaster(str(excel_path), verbose=False)
        used_sheet_names: set[str] = set()

        self._write_chart_sheet(em, tables, used_sheet_names)
        for name, df in tables.items():
            sheet_name = self._excel_sheet_name(name, used_sheet_names)
            ws = em.add_worksheet(sheet_name, zoom_perc=90)
            em.write_dataframe(ws, df=df, title=name, index=False)
        em.close_workbook()

    def _write_chart_sheet(
        self,
        em: Any,
        tables: dict[str, pd.DataFrame],
        used_sheet_names: set[str],
    ) -> None:
        ws = em.add_worksheet(self._excel_sheet_name("Charts", used_sheet_names), zoom_perc=90)
        em.merge_col(ws, loc=(0, 0), ncols=8, text="Sample Analysis Charts", cformat="BLUE_H4")

        coverage_chart = self._label_coverage_chart_data(tables["label_coverage_summary"])
        rec_chart = self._recommendation_chart_data(tables["split_recommendation"])
        oot_chart = self._oot_window_chart_data(tables["split_candidate_summary"])
        seed_chart = self._seed_stability_chart_data(tables["split_candidate_summary"])
        segment_chart = self._segment_chart_data(tables["segment_bad_rate_summary"])

        if not coverage_chart.empty:
            em.write_duo_chart(
                worksheet=ws,
                df=coverage_chart,
                x="label",
                y1_list=["n_observed"],
                y2_list=["observed_rate", "bad_rate"],
                c1_type="column",
                c2_type="line",
                y1_axis_range=(0, None),
                y2_axis_range=(0, None),
                y2_num_format="0.0%",
                loc=(2, 0),
                title="Label maturity coverage and observed bad rate",
                chart_size=(18, 13),
                xy_axes_name=("Target label", "Observed N", "Rate"),
                major_gridlines=False,
            )

        if not rec_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=rec_chart,
                x="label",
                y_list=["bad_rate_ins", "bad_rate_oos", "bad_rate_oot"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(2, 15),
                title="Recommended split bad-rate comparison",
                chart_size=(18, 13),
                xy_axes_name=("Target label", "Bad rate"),
                major_gridlines=True,
            )

        if not oot_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=oot_chart,
                x="oot_window_periods",
                y_list=[col for col in oot_chart.columns if col != "oot_window_periods"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(23, 0),
                title="Average max bad-rate gap by OOT window",
                chart_size=(18, 13),
                xy_axes_name=("OOT window periods", "Avg max bad-rate gap"),
                major_gridlines=True,
            )

        if not seed_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=seed_chart,
                x="seed",
                y_list=[col for col in seed_chart.columns if col != "seed"],
                chart_type="line",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(23, 15),
                title="Seed stability: average max bad-rate gap",
                chart_size=(18, 13),
                xy_axes_name=("Random seed", "Avg max bad-rate gap"),
                major_gridlines=True,
            )

        if not segment_chart.empty:
            em.write_chart(
                worksheet=ws,
                df=segment_chart,
                x="segment",
                y_list=[col for col in segment_chart.columns if col != "segment"],
                chart_type="column",
                y_axis_range=(0, None),
                y_num_format="0.0%",
                loc=(44, 0),
                title="Population bad-rate snapshot",
                chart_size=(18, 13),
                xy_axes_name=("Segment", "Bad rate"),
                major_gridlines=True,
            )

        em.write_dataframe(
            ws,
            df=tables["split_recommendation"],
            loc=(44, 15),
            title="Recommended split details",
            index=False,
        )

    def _excel_sheet_name(self, name: str, used_sheet_names: set[str]) -> str:
        clean = str(name).replace("/", "_").replace("\\", "_").replace(":", "-")
        clean = clean.replace("[", "(").replace("]", ")").replace("*", "_").replace("?", "_")
        clean = clean.strip("' ") or "Sheet"
        base = clean[:31]
        candidate = base
        i = 2
        while candidate.lower() in used_sheet_names:
            suffix = f"_{i}"
            candidate = f"{base[:31 - len(suffix)]}{suffix}"
            i += 1
        used_sheet_names.add(candidate.lower())
        return candidate

    def _label_coverage_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["target_col", "n_observed", "observed_rate", "bad_rate"]
        if df.empty or not set(cols).issubset(df.columns):
            return pd.DataFrame(columns=["label", "n_observed", "observed_rate", "bad_rate"])
        out = df[cols].copy()
        out["label"] = out["target_col"].map(self._short_target_name)
        return out[["label", "n_observed", "observed_rate", "bad_rate"]]

    def _recommendation_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["target_col", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"]
        if df.empty or not set(cols).issubset(df.columns):
            return pd.DataFrame(columns=["label", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"])
        out = df[cols].copy()
        out["label"] = out["target_col"].map(self._short_target_name)
        return out[["label", "bad_rate_ins", "bad_rate_oos", "bad_rate_oot"]]

    def _oot_window_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["oot_window_periods"])
        grouped = (
            df.groupby(["oot_window_periods", "target_col"], dropna=False)["max_abs_bad_rate_gap"]
            .mean()
            .reset_index()
        )
        grouped["target_col"] = grouped["target_col"].map(self._short_target_name)
        out = grouped.pivot(index="oot_window_periods", columns="target_col", values="max_abs_bad_rate_gap")
        return out.reset_index().sort_values("oot_window_periods")

    def _seed_stability_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["seed"])
        grouped = (
            df.groupby(["seed", "target_col"], dropna=False)["max_abs_bad_rate_gap"]
            .mean()
            .reset_index()
        )
        grouped["target_col"] = grouped["target_col"].map(self._short_target_name)
        out = grouped.pivot(index="seed", columns="target_col", values="max_abs_bad_rate_gap")
        return out.reset_index().sort_values("seed")

    def _segment_chart_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["segment"])
        sub = df[
            (df["group_type"] == "population")
            & (df["group_cols"].isin(self.config.population_dims))
        ].copy()
        if sub.empty:
            return pd.DataFrame(columns=["segment"])

        top_segments = (
            sub.groupby("group_value", dropna=False)["n"].sum().sort_values(ascending=False).head(10).index
        )
        sub = sub[sub["group_value"].isin(top_segments)]
        sub["target_col"] = sub["target_col"].map(self._short_target_name)
        out = sub.pivot_table(
            index="group_value",
            columns="target_col",
            values="bad_rate",
            aggfunc="mean",
        )
        return out.reset_index().rename(columns={"group_value": "segment"})

    def _short_target_name(self, target_col: str) -> str:
        return target_col.replace("y_flag_dpd7_in_", "").upper()
