from __future__ import annotations

import math
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .ODPS_Tool import ODPSRunner
from .Parallel_Engine import ParallelApplyConfig, ParallelApplyEngine
from .utils import parse_sql_file


Backend = Literal["thread", "process", "sequential"]
WriteMode = Literal["overwrite", "append"]
PullSplitStrategy = Literal["auto", "hash", "row_number"]


@dataclass
class ParallelODPSConfig:
    unique_key: str | None = None
    chunk_size: int | None = None
    n_chunks: int | None = None
    n_jobs: int = 3
    backend: Backend = "thread"
    pull_split_strategy: PullSplitStrategy = "auto"
    row_number_order_by: str | None = None
    row_number_col: str = "__smf_parallel_odps_rn__"
    validate_unique_key: bool = True
    chunk_filter_key: str = "chunk_filter"
    tmp_dir: Path = field(default_factory=lambda: Path("data/_chunks"))
    tmp_table_prefix: str = "tmp_parallel_odps"
    cleanup_tmp: bool = True
    keep_tmp_on_error: bool = False

    def __post_init__(self) -> None:
        self.tmp_dir = Path(self.tmp_dir)


def _get_runner(runner: ODPSRunner | None = None) -> ODPSRunner:
    return runner if runner is not None else ODPSRunner()


def _delete_table(runner: ODPSRunner, table_name: str) -> None:
    if hasattr(runner, "delete_table"):
        success = False
        try:
            runner.delete_table(table_name, if_exists=True)
        except TypeError:
            runner.delete_table(table_name)
        return
    odps_client = getattr(runner, "o", None)
    if odps_client is not None and hasattr(odps_client, "delete_table"):
        odps_client.delete_table(table_name, if_exists=True)
        return
    runner.run_sql(f"DROP TABLE IF EXISTS {table_name};", to_df=False)


def _strip_trailing_semicolon(sql: str) -> str:
    return sql.strip().rstrip(";").strip()


def _pull_one_chunk(
    chunk_spec: dict[str, Any],
    runner: ODPSRunner | None,
    tmp_dir_str: str,
) -> dict[str, Any]:
    active_runner = _get_runner(runner)
    chunk_id = int(chunk_spec["chunk"])
    rendered_sql = str(chunk_spec["sql"])
    drop_cols = list(chunk_spec.get("drop_cols") or [])

    df_chunk = active_runner.run_sql(rendered_sql, to_df=True, n_process=1)
    if drop_cols:
        df_chunk = df_chunk.drop(columns=[col for col in drop_cols if col in df_chunk.columns])
    chunk_path = Path(tmp_dir_str) / f"_pull_chunk_{chunk_id:04d}.csv"
    df_chunk.to_csv(chunk_path, index=False)
    n_rows = len(df_chunk)
    del df_chunk

    return {"chunk": chunk_id, "rows": n_rows, "path": str(chunk_path)}


def _push_one_chunk(
    chunk_spec: dict[str, Any],
    runner: ODPSRunner | None,
) -> dict[str, Any]:
    active_runner = _get_runner(runner)
    chunk_id = int(chunk_spec["chunk"])
    tmp_table = str(chunk_spec["tmp_table"])
    local_csv_path = chunk_spec.get("csv_path")

    if local_csv_path:
        df_chunk = pd.read_csv(local_csv_path)
    else:
        df_chunk = chunk_spec["data"]

    active_runner.upload_df(df_chunk, tmp_table)
    return {"chunk": chunk_id, "rows": len(df_chunk), "tmp_table": tmp_table, "csv_path": local_csv_path}


class ParallelODPSManager:
    """Parallel ODPS pull/push helper built on ODPSRunner and ParallelApplyEngine."""

    _VALID_BACKENDS = {"thread", "process", "sequential"}
    _VALID_WRITE_MODES = {"overwrite", "append"}
    _VALID_PULL_STRATEGIES = {"auto", "hash", "row_number"}

    def __init__(self, config: ParallelODPSConfig, odps_runner: ODPSRunner | None = None):
        self.config = config
        self.odps_runner = odps_runner or ODPSRunner()
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.config
        if cfg.backend not in self._VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(self._VALID_BACKENDS)}")
        if cfg.pull_split_strategy not in self._VALID_PULL_STRATEGIES:
            raise ValueError(f"pull_split_strategy must be one of {sorted(self._VALID_PULL_STRATEGIES)}")
        if cfg.chunk_size is not None and cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if cfg.n_chunks is not None and cfg.n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        if cfg.chunk_size is not None and cfg.n_chunks is not None:
            raise ValueError("chunk_size and n_chunks cannot be used together.")
        if cfg.n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer.")
        if not cfg.chunk_filter_key:
            raise ValueError("chunk_filter_key cannot be empty.")
        if not cfg.row_number_col:
            raise ValueError("row_number_col cannot be empty.")
        if not cfg.tmp_table_prefix:
            raise ValueError("tmp_table_prefix cannot be empty.")

    def _runner_for_backend(self) -> ODPSRunner | None:
        return None if self.config.backend == "process" else self.odps_runner

    def _auto_count_query(self, sql_path: str, template_kwargs: dict[str, Any]) -> str:
        kwargs = dict(template_kwargs)
        kwargs[self.config.chunk_filter_key] = "1=1"
        rendered_sql = _strip_trailing_semicolon(parse_sql_file(sql_path=sql_path, **kwargs))
        return f"SELECT COUNT(1) FROM ({rendered_sql}) __count_src;"

    def _assert_chunk_filter_placeholder(self, sql_path: str, template_kwargs: dict[str, Any]) -> None:
        sentinel = "__SMF_CHUNK_FILTER_SENTINEL__=1"
        kwargs = dict(template_kwargs)
        kwargs[self.config.chunk_filter_key] = sentinel
        rendered_sql = parse_sql_file(sql_path=sql_path, **kwargs)
        if sentinel not in rendered_sql:
            raise ValueError(
                f"pull SQL template must contain {{{self.config.chunk_filter_key}}}"
            )

    def _render_pull_sql(self, sql_path: str, template_kwargs: dict[str, Any], chunk_filter: str) -> str:
        kwargs = dict(template_kwargs)
        kwargs[self.config.chunk_filter_key] = chunk_filter
        return parse_sql_file(sql_path=sql_path, **kwargs)

    def _resolve_pull_strategy(self) -> str:
        cfg = self.config
        strategy = cfg.pull_split_strategy
        if strategy == "auto":
            return "hash" if cfg.unique_key else "row_number"
        if strategy == "hash" and not cfg.unique_key:
            raise ValueError("unique_key is required when pull_split_strategy='hash'.")
        return strategy

    def _resolve_pull_n_chunks(
        self,
        count_query: str | None,
        sql_path: str,
        template_kwargs: dict[str, Any],
        strategy: str,
        staging_table: str | None = None,
    ) -> int:
        cfg = self.config
        if cfg.n_chunks is not None:
            return cfg.n_chunks
        if cfg.chunk_size is None:
            raise ValueError("chunk_size or n_chunks is required for pull().")
        if count_query is None:
            if strategy == "row_number":
                if not staging_table:
                    raise ValueError("staging_table is required to count row_number pull chunks.")
                count_query = f"SELECT COUNT(1) FROM {staging_table};"
            else:
                count_query = self._auto_count_query(sql_path, template_kwargs)
        count_df = self.odps_runner.run_sql(count_query, to_df=True)
        total_rows = int(count_df.iloc[0, 0])
        return max(1, math.ceil(total_rows / cfg.chunk_size))

    def _hash_chunk_filter(self, chunk_id: int, n_chunks: int) -> str:
        return f"ABS(HASH({self.config.unique_key})) % {n_chunks} = {chunk_id}"

    def _validate_hash_pull_sql(
        self,
        sql_path: str,
        template_kwargs: dict[str, Any],
        n_chunks: int,
    ) -> None:
        if not self.config.validate_unique_key:
            return
        rendered_sql = self._render_pull_sql(
            sql_path=sql_path,
            template_kwargs=template_kwargs,
            chunk_filter=self._hash_chunk_filter(0, n_chunks),
        )
        probe_sql = (
            "SELECT * FROM (\n"
            f"{_strip_trailing_semicolon(rendered_sql)}\n"
            ") __smf_unique_key_probe LIMIT 1;"
        )
        try:
            self.odps_runner.run_sql(probe_sql, to_df=True, n_process=1)
        except Exception as exc:
            raise ValueError(
                f"unique_key validation failed for pull SQL. "
                f"Check that unique_key={self.config.unique_key!r} is visible in the SQL scope."
            ) from exc

    def _build_hash_pull_chunks(
        self,
        sql_path: str,
        template_kwargs: dict[str, Any],
        n_chunks: int,
    ) -> list[dict[str, Any]]:
        return [
            {
                "chunk": chunk_id,
                "sql": self._render_pull_sql(
                    sql_path=sql_path,
                    template_kwargs=template_kwargs,
                    chunk_filter=self._hash_chunk_filter(chunk_id, n_chunks),
                ),
            }
            for chunk_id in range(n_chunks)
        ]

    def _row_number_staging_table_name(self, run_id: str) -> str:
        return f"{self.config.tmp_table_prefix}_{run_id}_pull_stage"

    def _create_row_number_staging_table(
        self,
        sql_path: str,
        template_kwargs: dict[str, Any],
        staging_table: str,
    ) -> str:
        cfg = self.config
        base_sql = _strip_trailing_semicolon(
            self._render_pull_sql(
                sql_path=sql_path,
                template_kwargs=template_kwargs,
                chunk_filter="1=1",
            )
        )
        order_by = cfg.row_number_order_by or "1"
        create_sql = (
            f"CREATE TABLE {staging_table} AS\n"
            "SELECT\n"
            f"  ROW_NUMBER() OVER (ORDER BY {order_by}) AS {cfg.row_number_col},\n"
            "  __smf_base.*\n"
            "FROM (\n"
            f"{base_sql}\n"
            ") __smf_base;"
        )
        self.odps_runner.run_sql(create_sql, to_df=False)
        return create_sql

    def _build_row_number_pull_chunks(self, staging_table: str, n_chunks: int) -> list[dict[str, Any]]:
        row_number_col = self.config.row_number_col
        return [
            {
                "chunk": chunk_id,
                "sql": (
                    f"SELECT *\n"
                    f"FROM {staging_table}\n"
                    f"WHERE ({row_number_col} - 1) % {n_chunks} = {chunk_id};"
                ),
                "drop_cols": [row_number_col],
            }
            for chunk_id in range(n_chunks)
        ]

    def _run_pull_chunks(self, chunk_specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cfg = self.config
        engine_cfg = ParallelApplyConfig(
            split_axis="chunk",
            backend=cfg.backend,
            n_jobs=cfg.n_jobs,
            combine="list",
            on_error="collect",
        )
        result = ParallelApplyEngine(engine_cfg).run(
            func=_pull_one_chunk,
            chunks=chunk_specs,
            func_args=(
                self._runner_for_backend(),
                str(cfg.tmp_dir),
            ),
        )
        if len(result.errors):
            raise RuntimeError(f"{len(result.errors)}/{len(chunk_specs)} ODPS pull chunks failed:\n{result.errors}")
        return sorted(result.output, key=lambda item: item["chunk"])

    def pull(
        self,
        sql_path: str,
        out_path: str,
        count_query: str | None = None,
        **template_kwargs: Any,
    ) -> dict[str, Any]:
        cfg = self.config
        self._assert_chunk_filter_placeholder(sql_path, template_kwargs)
        cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
        strategy = self._resolve_pull_strategy()
        staging_table: str | None = None
        success = False

        try:
            if strategy == "row_number":
                run_id = uuid.uuid4().hex[:12]
                staging_table = self._row_number_staging_table_name(run_id)
                self._create_row_number_staging_table(sql_path, template_kwargs, staging_table)
                n_chunks = self._resolve_pull_n_chunks(
                    count_query=count_query,
                    sql_path=sql_path,
                    template_kwargs=template_kwargs,
                    strategy=strategy,
                    staging_table=staging_table,
                )
                chunk_specs = self._build_row_number_pull_chunks(staging_table, n_chunks)
            else:
                n_chunks = self._resolve_pull_n_chunks(
                    count_query=count_query,
                    sql_path=sql_path,
                    template_kwargs=template_kwargs,
                    strategy=strategy,
                )
                self._validate_hash_pull_sql(sql_path, template_kwargs, n_chunks)
                chunk_specs = self._build_hash_pull_chunks(sql_path, template_kwargs, n_chunks)

            chunk_summaries = self._run_pull_chunks(chunk_specs)
            final_path = Path(out_path)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.unlink(missing_ok=True)

            with open(final_path, "wb") as fout:
                for idx, summary in enumerate(chunk_summaries):
                    chunk_file = Path(summary["path"])
                    with open(chunk_file, "rb") as fin:
                        if idx > 0:
                            fin.readline()
                        shutil.copyfileobj(fin, fout)
                    chunk_file.unlink()

            success = True
            return {
                "pull_strategy": strategy,
                "staging_table": staging_table,
                "n_chunks": n_chunks,
                "total_rows": sum(item["rows"] for item in chunk_summaries),
                "out_path": str(final_path),
                "per_chunk_rows": [item["rows"] for item in chunk_summaries],
            }
        finally:
            if (
                staging_table
                and cfg.cleanup_tmp
                and (success or not cfg.keep_tmp_on_error)
            ):
                _delete_table(self.odps_runner, staging_table)

    def push(
        self,
        data: pd.DataFrame | str | Path,
        target_table: str,
        write_mode: WriteMode | str | None = None,
    ) -> dict[str, Any]:
        if write_mode not in self._VALID_WRITE_MODES:
            raise ValueError("write_mode is required and must be one of ['append', 'overwrite'].")
        if not target_table:
            raise ValueError("target_table cannot be empty.")

        cfg = self.config
        cfg.tmp_dir.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex[:12]
        chunk_specs, local_temp_paths = self._build_push_chunks(data=data, run_id=run_id)
        if not chunk_specs:
            raise ValueError("push data produced no chunks.")

        tmp_tables = [spec["tmp_table"] for spec in chunk_specs]
        success = False
        try:
            engine_cfg = ParallelApplyConfig(
                split_axis="chunk",
                backend=cfg.backend,
                n_jobs=cfg.n_jobs,
                combine="list",
                on_error="collect",
            )
            result = ParallelApplyEngine(engine_cfg).run(
                func=_push_one_chunk,
                chunks=chunk_specs,
                func_args=(self._runner_for_backend(),),
            )
            if len(result.errors):
                raise RuntimeError(f"{len(result.errors)}/{len(chunk_specs)} ODPS push chunks failed:\n{result.errors}")

            chunk_summaries = sorted(result.output, key=lambda item: item["chunk"])
            union_sql = self._build_union_sql(tmp_tables)
            final_sql = self._write_final_table(target_table=target_table, union_sql=union_sql, write_mode=str(write_mode))

            success = True
            return {
                "n_chunks": len(chunk_summaries),
                "total_rows": sum(item["rows"] for item in chunk_summaries),
                "target_table": target_table,
                "write_mode": write_mode,
                "tmp_tables": tmp_tables,
                "per_chunk_rows": [item["rows"] for item in chunk_summaries],
                "union_sql": union_sql,
                "final_sql": final_sql,
            }
        finally:
            self._cleanup_local_files(local_temp_paths)
            if cfg.cleanup_tmp and (success or not cfg.keep_tmp_on_error):
                self._cleanup_tmp_tables(tmp_tables)

    def _build_push_chunks(self, data: pd.DataFrame | str | Path, run_id: str) -> tuple[list[dict[str, Any]], list[Path]]:
        if isinstance(data, pd.DataFrame):
            return self._build_dataframe_push_chunks(data, run_id), []
        if isinstance(data, (str, Path)):
            return self._build_csv_push_chunks(Path(data), run_id)
        raise TypeError("data must be a pandas DataFrame or a CSV path.")

    def _build_dataframe_push_chunks(self, data: pd.DataFrame, run_id: str) -> list[dict[str, Any]]:
        if len(data.columns) == 0:
            raise ValueError("data must contain at least one column.")
        positions = self._split_positions(len(data))
        specs = []
        for chunk_id, pos in enumerate(positions):
            chunk_df = data.iloc[pos].copy() if len(pos) else data.iloc[0:0].copy()
            specs.append({
                "chunk": chunk_id,
                "data": chunk_df,
                "tmp_table": self._tmp_table_name(run_id, chunk_id),
            })
        return specs

    def _build_csv_push_chunks(self, csv_path: Path, run_id: str) -> tuple[list[dict[str, Any]], list[Path]]:
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        chunk_size = self._resolve_csv_chunk_size(csv_path)
        specs: list[dict[str, Any]] = []
        local_paths: list[Path] = []
        for chunk_id, chunk_df in enumerate(pd.read_csv(csv_path, chunksize=chunk_size)):
            local_path = self.config.tmp_dir / f"_push_{run_id}_{chunk_id:04d}.csv"
            chunk_df.to_csv(local_path, index=False)
            local_paths.append(local_path)
            specs.append({
                "chunk": chunk_id,
                "csv_path": str(local_path),
                "tmp_table": self._tmp_table_name(run_id, chunk_id),
            })
        return specs, local_paths

    def _resolve_csv_chunk_size(self, csv_path: Path) -> int:
        cfg = self.config
        if cfg.chunk_size is not None:
            return cfg.chunk_size
        if cfg.n_chunks is None:
            raise ValueError("chunk_size or n_chunks is required for CSV push().")
        with open(csv_path, "rb") as fin:
            n_lines = sum(1 for _ in fin)
        n_rows = max(0, n_lines - 1)
        return max(1, math.ceil(n_rows / cfg.n_chunks))

    def _split_positions(self, n_rows: int) -> list[np.ndarray]:
        cfg = self.config
        if n_rows < 0:
            raise ValueError("n_rows must be non-negative.")
        if n_rows == 0:
            return [np.array([], dtype=int)]
        if cfg.chunk_size is not None:
            n_chunks = math.ceil(n_rows / cfg.chunk_size)
        elif cfg.n_chunks is not None:
            n_chunks = min(cfg.n_chunks, n_rows)
        else:
            raise ValueError("chunk_size or n_chunks is required for push().")
        return [arr for arr in np.array_split(np.arange(n_rows), n_chunks) if len(arr) > 0]

    def _tmp_table_name(self, run_id: str, chunk_id: int) -> str:
        return f"{self.config.tmp_table_prefix}_{run_id}_{chunk_id:04d}"

    @staticmethod
    def _build_union_sql(tmp_tables: list[str]) -> str:
        return "\nUNION ALL\n".join(f"SELECT * FROM {table_name}" for table_name in tmp_tables)

    def _write_final_table(self, target_table: str, union_sql: str, write_mode: str) -> str:
        if write_mode == "overwrite":
            _delete_table(self.odps_runner, target_table)
            final_sql = f"CREATE TABLE {target_table} AS\n{union_sql};"
        else:
            final_sql = f"INSERT INTO TABLE {target_table}\n{union_sql};"
        self.odps_runner.run_sql(final_sql, to_df=False)
        return final_sql

    def _cleanup_tmp_tables(self, tmp_tables: list[str]) -> None:
        for table_name in tmp_tables:
            _delete_table(self.odps_runner, table_name)

    @staticmethod
    def _cleanup_local_files(paths: list[Path]) -> None:
        for path in paths:
            path.unlink(missing_ok=True)


ParallelODPSPuller = ParallelODPSManager
