from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


_BUSINESS_TYPES = [
    "basic_info",
    "multi_loan",
    "credit_report_stats",
    "historical_limit",
    "overdue_status",
    "query_count",
    "telecom_data",
    "consumption_data",
    "income_data",
    "address_data",
]


@dataclass
class MockSamplePipelineConfig:
    n_samples: int = 80_000
    applied_sample: int = 1
    approve_rate: float = 0.25
    num_online_scores: int = 5
    y_flag_candidates: list[int] = field(default_factory=lambda: [15, 30, 45])
    num_features: int = 20
    min_num_feature_business_type: int = 5
    random_state: int = 42
    observation_timestamp: str | pd.Timestamp | None = None
    application_months: int = 18
    write_csv: bool = False
    output_path: str = "output/mock_sample/mock_sample.csv"


@dataclass
class MockSamplePipelineResult:
    data: pd.DataFrame
    summary: pd.DataFrame
    feature_metadata: pd.DataFrame
    output_path: str | None


class MockSamplePipeline:
    """Generate synthetic application or approved samples for SMF demos."""

    def __init__(self, config: MockSamplePipelineConfig | None = None):
        self.config = config or MockSamplePipelineConfig()

    def run(self) -> MockSamplePipelineResult:
        cfg = self.config
        self._validate_config()
        rng = np.random.default_rng(cfg.random_state)
        observation_ts = self._observation_timestamp()

        data = self._base_applications(rng, observation_ts)
        latent_risk = self._latent_risk(rng, data)
        data["is_approved"] = self._approval_flag(rng, latent_risk)
        feature_metadata = self._add_features(rng, data, latent_risk)
        self._add_online_scores(rng, data, latent_risk)
        self._add_y_flags(rng, data, latent_risk, observation_ts)

        if cfg.applied_sample == 0:
            data = data[data["is_approved"].eq(1)].reset_index(drop=True)

        summary = self._summary(data)
        output_path = self._write_csv(data)
        return MockSamplePipelineResult(
            data=data,
            summary=summary,
            feature_metadata=feature_metadata,
            output_path=output_path,
        )

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if cfg.applied_sample not in {0, 1}:
            raise ValueError("applied_sample must be 1 for full applications or 0 for approved samples")
        if not 0 < cfg.approve_rate < 1:
            raise ValueError("approve_rate must be between 0 and 1")
        if cfg.num_online_scores < 0:
            raise ValueError("num_online_scores must be non-negative")
        if cfg.num_features < 0:
            raise ValueError("num_features must be non-negative")
        if cfg.application_months <= 0:
            raise ValueError("application_months must be positive")
        if not cfg.y_flag_candidates:
            raise ValueError("y_flag_candidates must not be empty")
        if any(int(days) <= 0 for days in cfg.y_flag_candidates):
            raise ValueError("y_flag_candidates must contain positive day values")
        if cfg.num_features == 0:
            if cfg.min_num_feature_business_type != 0:
                raise ValueError("min_num_feature_business_type must be 0 when num_features is 0")
        elif not 1 <= cfg.min_num_feature_business_type <= min(cfg.num_features, len(_BUSINESS_TYPES)):
            raise ValueError(
                "min_num_feature_business_type must be between 1 and min(num_features, 10)"
            )

    def _observation_timestamp(self) -> pd.Timestamp:
        if self.config.observation_timestamp is None:
            return pd.Timestamp.today().normalize()
        return pd.Timestamp(self.config.observation_timestamp)

    def _base_applications(self, rng: np.random.Generator, observation_ts: pd.Timestamp) -> pd.DataFrame:
        cfg = self.config
        start_ts = observation_ts - pd.DateOffset(months=cfg.application_months)
        total_seconds = max(int((observation_ts - start_ts).total_seconds()), 1)
        offsets = rng.integers(0, total_seconds + 1, size=cfg.n_samples)
        apply_ts = start_ts + pd.to_timedelta(offsets, unit="s")
        data = pd.DataFrame(
            {
                "flow_id": [f"FLOW_{i:08d}" for i in range(cfg.n_samples)],
                "apply_timestamp": apply_ts,
            }
        )
        data["apply_week"] = data["apply_timestamp"].dt.strftime("%G-W%V")
        data["apply_month"] = data["apply_timestamp"].dt.strftime("%Y-%m")
        data["apply_quarter"] = data["apply_timestamp"].dt.to_period("Q").astype(str)
        return data

    def _latent_risk(self, rng: np.random.Generator, data: pd.DataFrame) -> np.ndarray:
        hour = data["apply_timestamp"].dt.hour.to_numpy()
        month = data["apply_timestamp"].dt.month.to_numpy()
        return (
            0.10 * np.sin(hour / 24 * 2 * np.pi)
            + 0.08 * np.cos(month / 12 * 2 * np.pi)
            + rng.normal(0, 1, size=len(data))
        )

    def _approval_flag(self, rng: np.random.Generator, latent_risk: np.ndarray) -> np.ndarray:
        n_approved = int(round(len(latent_risk) * self.config.approve_rate))
        n_approved = min(max(n_approved, 1), len(latent_risk) - 1)
        approval_score = latent_risk + rng.normal(0, 0.25, size=len(latent_risk))
        approved_idx = np.argsort(approval_score)[:n_approved]
        approved = np.zeros(len(latent_risk), dtype=int)
        approved[approved_idx] = 1
        return approved

    def _add_features(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        latent_risk: np.ndarray,
    ) -> pd.DataFrame:
        cfg = self.config
        rows = []
        if cfg.num_features == 0:
            return pd.DataFrame(columns=["feature", "business_type", "distribution"])

        n_types = min(max(cfg.min_num_feature_business_type, min(cfg.num_features, len(_BUSINESS_TYPES))), len(_BUSINESS_TYPES))
        chosen_types = list(rng.choice(_BUSINESS_TYPES, size=n_types, replace=False))
        assigned_types = chosen_types.copy()
        if cfg.num_features > n_types:
            assigned_types.extend(rng.choice(chosen_types, size=cfg.num_features - n_types, replace=True).tolist())
        rng.shuffle(assigned_types)

        type_counter: dict[str, int] = {}
        for business_type in assigned_types:
            type_counter[business_type] = type_counter.get(business_type, 0) + 1
            idx = type_counter[business_type]
            col = f"feat_{business_type}_{idx:02d}"
            distribution = self._feature_distribution(business_type)
            if distribution == "normal":
                data[col] = np.round(latent_risk * rng.uniform(0.1, 0.4) + rng.normal(0, 1, len(data)), 6)
            elif distribution == "count":
                rate = np.clip(np.exp(0.35 * latent_risk + rng.normal(0, 0.15, len(data))), 0.05, 20)
                data[col] = rng.poisson(rate)
            elif distribution == "binary":
                prob = 1 / (1 + np.exp(-(0.4 * latent_risk + rng.normal(0, 0.5, len(data)))))
                data[col] = (rng.random(len(data)) < prob).astype(int)
            else:
                raw = 1 / (1 + np.exp(-(0.5 * latent_risk + rng.normal(0, 0.7, len(data)))))
                data[col] = np.round(raw, 6)
            rows.append({"feature": col, "business_type": business_type, "distribution": distribution})
        return pd.DataFrame(rows)

    def _feature_distribution(self, business_type: str) -> str:
        if business_type in {"multi_loan", "query_count", "overdue_status"}:
            return "count"
        if business_type in {"basic_info", "address_data"}:
            return "binary"
        if business_type in {"telecom_data", "consumption_data"}:
            return "ratio"
        return "normal"

    def _add_online_scores(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        latent_risk: np.ndarray,
    ) -> None:
        for i in range(1, self.config.num_online_scores + 1):
            shift = rng.normal(0, 0.08)
            noise = rng.normal(0, 0.45, len(data))
            score = 1 / (1 + np.exp(-(latent_risk + shift + noise)))
            data[f"online_model_pb_{i}"] = np.round(score, 6)

    def _add_y_flags(
        self,
        rng: np.random.Generator,
        data: pd.DataFrame,
        latent_risk: np.ndarray,
        observation_ts: pd.Timestamp,
    ) -> None:
        approved = data["is_approved"].to_numpy() == 1
        base_bad = 1 / (1 + np.exp(-(latent_risk - 1.35)))
        for days in sorted({int(x) for x in self.config.y_flag_candidates}):
            col = f"y_flag_dpd7_in_{days}d"
            matured = approved & ((data["apply_timestamp"] + pd.to_timedelta(days, unit="D")) <= observation_ts).to_numpy()
            y = np.full(len(data), np.nan)
            multiplier = 0.65 + min(days, 180) / 180
            prob = np.clip(base_bad * multiplier, 0.003, 0.85)
            y[matured] = (rng.random(matured.sum()) < prob[matured]).astype(float)
            data[col] = y

        self._validate_y_maturity(data)

    def _validate_y_maturity(self, data: pd.DataFrame) -> None:
        cols = [f"y_flag_dpd7_in_{days}d" for days in sorted({int(x) for x in self.config.y_flag_candidates})]
        observed_counts = [int(data[col].notna().sum()) for col in cols]
        if observed_counts != sorted(observed_counts, reverse=True) or len(set(observed_counts)) != len(observed_counts):
            raise ValueError(
                "Observed y-label counts should strictly decrease for longer horizons: "
                f"{dict(zip(cols, observed_counts))}"
            )

    def _summary(self, data: pd.DataFrame) -> pd.DataFrame:
        rows = [
            {"metric": "n_rows", "value": len(data)},
            {"metric": "applied_sample", "value": self.config.applied_sample},
            {"metric": "approval_rate", "value": float(data["is_approved"].mean())},
            {"metric": "num_online_scores", "value": self.config.num_online_scores},
            {"metric": "num_features", "value": self.config.num_features},
        ]
        for days in sorted({int(x) for x in self.config.y_flag_candidates}):
            col = f"y_flag_dpd7_in_{days}d"
            observed = data[col].notna()
            rows.extend(
                [
                    {"metric": f"{col}_observed_n", "value": int(observed.sum())},
                    {"metric": f"{col}_observed_rate", "value": float(observed.mean())},
                    {"metric": f"{col}_bad_rate", "value": float(data[col].mean())},
                ]
            )
        return pd.DataFrame(rows)

    def _write_csv(self, data: pd.DataFrame) -> str | None:
        if not self.config.write_csv:
            return None
        path = Path(self.config.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return str(path.resolve())
