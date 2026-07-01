from __future__ import annotations

import math
import multiprocessing
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

import numpy as np
import pandas as pd


SplitAxis = Literal["row", "column", "chunk", "auto"]
Backend = Literal["process", "thread", "sequential"]
CombineMode = Literal["concat", "list", "dict", "none"]
ErrorMode = Literal["raise", "collect"]


@dataclass
class ParallelApplyConfig:
    split_axis: SplitAxis = "row"
    backend: Backend = "process"
    n_jobs: int | str | None = "auto"
    chunk_size: int | None = None
    n_chunks: int | None = None
    preserve_order: bool = True
    combine: CombineMode = "concat"
    concat_axis: Literal[0, 1] | None = None
    pass_chunk_info: bool = False
    timeout: float | None = None
    on_error: ErrorMode = "raise"
    validate_picklable: bool = True
    auto_probe: bool = False
    random_state: int | None = None
    required_cols: list[str] = field(default_factory=list)
    id_cols: list[str] = field(default_factory=list)


@dataclass
class ParallelApplyResult:
    output: Any
    chunk_outputs: list[Any] = field(default_factory=list)
    errors: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: dict[str, Any] = field(default_factory=dict)
    config: ParallelApplyConfig | None = None
    split_axis_resolved: str | None = None


@dataclass
class _ChunkSpec:
    chunk_id: int
    chunk: Any
    rows: int | None = None
    columns: list[str] | None = None


def _run_chunk(
    func: Callable[..., Any],
    chunk: Any,
    func_args: tuple[Any, ...],
    func_kwargs: dict[str, Any],
    pass_chunk_info: bool,
    chunk_info: dict[str, Any],
    collect_errors: bool,
) -> dict[str, Any]:
    try:
        kwargs = dict(func_kwargs)
        if pass_chunk_info:
            kwargs["chunk_info"] = chunk_info
        return {
            "ok": True,
            "chunk_id": chunk_info["chunk_id"],
            "output": func(chunk, *func_args, **kwargs),
            "error": None,
        }
    except Exception as exc:
        if not collect_errors:
            raise
        return {
            "ok": False,
            "chunk_id": chunk_info["chunk_id"],
            "output": None,
            "error": {
                "chunk_id": chunk_info["chunk_id"],
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }


class ParallelApplyEngine:
    """Parallel function dispatcher for DataFrame and custom chunk workloads."""

    _VALID_SPLIT_AXIS = {"row", "column", "chunk", "auto"}
    _VALID_BACKENDS = {"process", "thread", "sequential"}
    _VALID_COMBINE = {"concat", "list", "dict", "none"}
    _VALID_ON_ERROR = {"raise", "collect"}

    def __init__(self, config: ParallelApplyConfig | None = None):
        self.config = config or ParallelApplyConfig()
        self._validate_config(self.config)

    def run(
        self,
        data: Any = None,
        func: Callable[..., Any] | None = None,
        func_args: Sequence[Any] = (),
        func_kwargs: dict[str, Any] | None = None,
        chunks: Sequence[Any] | None = None,
    ) -> ParallelApplyResult:
        if func is None:
            raise ValueError("func is required.")

        cfg = self.config
        func_kwargs = {} if func_kwargs is None else dict(func_kwargs)
        func_args = tuple(func_args)
        self._validate_runtime_inputs(data=data, chunks=chunks)
        if cfg.validate_picklable and cfg.backend == "process":
            self._validate_picklable(func, func_args, func_kwargs)

        start = time.time()
        split_axis = self._resolve_split_axis(data, func, func_args, func_kwargs, chunks)
        chunk_specs = self._make_chunks(data=data, chunks=chunks, split_axis=split_axis)
        n_jobs = self._resolve_n_jobs(cfg.n_jobs, len(chunk_specs))

        raw_results = self._execute(
            func=func,
            func_args=func_args,
            func_kwargs=func_kwargs,
            chunk_specs=chunk_specs,
            n_jobs=n_jobs,
        )
        if cfg.preserve_order:
            raw_results = sorted(raw_results, key=lambda item: item["chunk_id"])

        successful = [item for item in raw_results if item["ok"]]
        chunk_outputs = [item["output"] for item in successful]
        errors = pd.DataFrame([item["error"] for item in raw_results if not item["ok"]])
        output = self._combine_outputs(successful, split_axis)
        elapsed = time.time() - start

        return ParallelApplyResult(
            output=output,
            chunk_outputs=chunk_outputs,
            errors=errors,
            summary={
                "split_axis": split_axis,
                "backend": cfg.backend,
                "n_jobs": n_jobs,
                "n_chunks": len(chunk_specs),
                "n_success": len(successful),
                "n_error": int(len(errors)),
                "elapsed_seconds": elapsed,
            },
            config=cfg,
            split_axis_resolved=split_axis,
        )

    def _validate_config(self, cfg: ParallelApplyConfig) -> None:
        if cfg.split_axis not in self._VALID_SPLIT_AXIS:
            raise ValueError(f"split_axis must be one of {sorted(self._VALID_SPLIT_AXIS)}")
        if cfg.backend not in self._VALID_BACKENDS:
            raise ValueError(f"backend must be one of {sorted(self._VALID_BACKENDS)}")
        if cfg.combine not in self._VALID_COMBINE:
            raise ValueError(f"combine must be one of {sorted(self._VALID_COMBINE)}")
        if cfg.on_error not in self._VALID_ON_ERROR:
            raise ValueError(f"on_error must be one of {sorted(self._VALID_ON_ERROR)}")
        if cfg.chunk_size is not None and cfg.chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer.")
        if cfg.n_chunks is not None and cfg.n_chunks <= 0:
            raise ValueError("n_chunks must be a positive integer.")
        if cfg.chunk_size is not None and cfg.n_chunks is not None:
            raise ValueError("chunk_size and n_chunks cannot be used together.")
        if cfg.concat_axis is not None and cfg.concat_axis not in {0, 1}:
            raise ValueError("concat_axis must be 0, 1, or None.")

    def _validate_runtime_inputs(self, data: Any, chunks: Sequence[Any] | None) -> None:
        cfg = self.config
        if cfg.split_axis in {"row", "column", "auto"} and not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame when split_axis is row, column, or auto.")
        if cfg.split_axis == "chunk" and chunks is None:
            raise ValueError("chunks is required when split_axis='chunk'.")

    def _validate_picklable(
        self,
        func: Callable[..., Any],
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
    ) -> None:
        try:
            from joblib.externals import cloudpickle

            cloudpickle.dumps((func, func_args, func_kwargs))
        except Exception as exc:
            raise TypeError(
                "func, func_args, and func_kwargs must be serializable for backend='process'. "
                "Use backend='thread' or backend='sequential' for non-serializable callables."
            ) from exc

    def _resolve_split_axis(
        self,
        data: Any,
        func: Callable[..., Any],
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
        chunks: Sequence[Any] | None,
    ) -> str:
        cfg = self.config
        if cfg.split_axis != "auto":
            return cfg.split_axis
        if chunks is not None:
            return "chunk"
        return self._probe_split_axis(data, func, func_args, func_kwargs)

    def _probe_split_axis(
        self,
        data: pd.DataFrame,
        func: Callable[..., Any],
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
    ) -> str:
        sample = data.sample(
            n=min(len(data), 20),
            random_state=self.config.random_state,
        ) if len(data) > 20 else data.copy()
        full_output = func(sample.copy(), *func_args, **func_kwargs)
        candidates: list[str] = []
        for axis in ("row", "column"):
            try:
                chunk_specs = self._make_chunks(data=sample, chunks=None, split_axis=axis, force_n_chunks=2)
                outputs = []
                for spec in chunk_specs:
                    outputs.append(func(spec.chunk, *func_args, **func_kwargs))
                probe_results = [
                    {"chunk_id": i, "output": output}
                    for i, output in enumerate(outputs)
                ]
                combined = self._combine_outputs(probe_results, axis)
                if self._outputs_equivalent(full_output, combined):
                    candidates.append(axis)
            except Exception:
                continue
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError(
                "split_axis='auto' is ambiguous: both row and column probes matched. "
                "Please set split_axis explicitly."
            )
        raise ValueError(
            "split_axis='auto' could not prove that the function is row- or column-splittable. "
            "Please set split_axis explicitly."
        )

    def _outputs_equivalent(self, left: Any, right: Any) -> bool:
        try:
            if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
                pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False)
                return True
            if isinstance(left, pd.Series) and isinstance(right, pd.Series):
                pd.testing.assert_series_equal(left, right, check_dtype=False, check_exact=False)
                return True
            if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
                return bool(np.allclose(left, right, equal_nan=True))
            return left == right
        except Exception:
            return False

    def _make_chunks(
        self,
        data: Any,
        chunks: Sequence[Any] | None,
        split_axis: str,
        force_n_chunks: int | None = None,
    ) -> list[_ChunkSpec]:
        if split_axis == "chunk":
            return [
                _ChunkSpec(chunk_id=i, chunk=chunk, rows=self._safe_len(chunk))
                for i, chunk in enumerate(chunks or [])
            ]
        if split_axis == "row":
            return self._make_row_chunks(data, force_n_chunks)
        if split_axis == "column":
            return self._make_column_chunks(data, force_n_chunks)
        raise ValueError(f"Unsupported split_axis: {split_axis}")

    def _make_row_chunks(self, data: pd.DataFrame, force_n_chunks: int | None = None) -> list[_ChunkSpec]:
        indices = self._split_positions(len(data), force_n_chunks)
        return [
            _ChunkSpec(chunk_id=i, chunk=data.iloc[pos].copy(), rows=len(pos), columns=list(data.columns))
            for i, pos in enumerate(indices)
            if len(pos) > 0
        ]

    def _make_column_chunks(self, data: pd.DataFrame, force_n_chunks: int | None = None) -> list[_ChunkSpec]:
        fixed_cols = self._dedupe(self.config.id_cols + self.config.required_cols)
        missing = [col for col in fixed_cols if col not in data.columns]
        if missing:
            raise KeyError(f"required/id columns not found in data: {missing}")
        split_cols = [col for col in data.columns if col not in set(fixed_cols)]
        positions = self._split_positions(len(split_cols), force_n_chunks)
        specs = []
        for i, pos in enumerate(positions):
            cols = fixed_cols + [split_cols[j] for j in pos]
            if len(cols) == 0:
                continue
            specs.append(
                _ChunkSpec(
                    chunk_id=i,
                    chunk=data.loc[:, cols].copy(),
                    rows=len(data),
                    columns=cols,
                )
            )
        return specs

    def _split_positions(self, n_items: int, force_n_chunks: int | None = None) -> list[np.ndarray]:
        if n_items < 0:
            raise ValueError("n_items must be non-negative.")
        if n_items == 0:
            return [np.array([], dtype=int)]
        if force_n_chunks is not None:
            n_chunks = min(max(1, force_n_chunks), n_items)
        elif self.config.chunk_size is not None:
            n_chunks = int(math.ceil(n_items / self.config.chunk_size))
        elif self.config.n_chunks is not None:
            n_chunks = min(self.config.n_chunks, n_items)
        else:
            n_chunks = min(self._resolve_n_jobs(self.config.n_jobs, n_items), n_items)
        return [arr for arr in np.array_split(np.arange(n_items), n_chunks) if len(arr) > 0]

    def _resolve_n_jobs(self, n_jobs: int | str | None, n_tasks: int) -> int:
        if n_tasks <= 0:
            return 1
        if n_jobs is None:
            resolved = 1
        elif isinstance(n_jobs, str):
            if n_jobs != "auto":
                raise ValueError("n_jobs as string only supports 'auto'.")
            resolved = max(1, multiprocessing.cpu_count() - 1)
        elif n_jobs == -1:
            resolved = multiprocessing.cpu_count()
        elif n_jobs <= 0:
            raise ValueError("n_jobs must be a positive integer, -1, 'auto', or None.")
        else:
            resolved = int(n_jobs)
        return min(max(1, resolved), max(1, n_tasks))

    def _execute(
        self,
        func: Callable[..., Any],
        func_args: tuple[Any, ...],
        func_kwargs: dict[str, Any],
        chunk_specs: list[_ChunkSpec],
        n_jobs: int,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        collect_errors = cfg.on_error == "collect"
        if cfg.backend == "sequential" or n_jobs == 1:
            return [
                _run_chunk(
                    func,
                    spec.chunk,
                    func_args,
                    func_kwargs,
                    cfg.pass_chunk_info,
                    self._chunk_info(spec),
                    collect_errors,
                )
                for spec in chunk_specs
            ]

        from joblib import Parallel, delayed

        backend = "threading" if cfg.backend == "thread" else "loky"
        parallel_kwargs = {"n_jobs": n_jobs, "backend": backend}
        if cfg.timeout is not None:
            parallel_kwargs["timeout"] = cfg.timeout
        return Parallel(**parallel_kwargs)(
            delayed(_run_chunk)(
                func,
                spec.chunk,
                func_args,
                func_kwargs,
                cfg.pass_chunk_info,
                self._chunk_info(spec),
                collect_errors,
            )
            for spec in chunk_specs
        )

    def _combine_outputs(self, successful_results: list[dict[str, Any]], split_axis: str) -> Any:
        cfg = self.config
        outputs = [item["output"] for item in successful_results]
        if cfg.combine == "none":
            return None
        if cfg.combine == "list":
            return outputs
        if cfg.combine == "dict":
            return {item["chunk_id"]: item["output"] for item in successful_results}
        if not outputs:
            return None
        concat_axis = cfg.concat_axis
        if concat_axis is None:
            concat_axis = 1 if split_axis == "column" else 0
        return pd.concat(outputs, axis=concat_axis)

    def _chunk_info(self, spec: _ChunkSpec) -> dict[str, Any]:
        return {
            "chunk_id": spec.chunk_id,
            "rows": spec.rows,
            "columns": spec.columns,
        }

    def _safe_len(self, obj: Any) -> int | None:
        try:
            return len(obj)
        except Exception:
            return None

    def _dedupe(self, values: Sequence[str]) -> list[str]:
        seen = set()
        result = []
        for value in values:
            if value not in seen:
                seen.add(value)
                result.append(value)
        return result


def parallel_apply(
    data: Any = None,
    func: Callable[..., Any] | None = None,
    func_args: Sequence[Any] = (),
    func_kwargs: dict[str, Any] | None = None,
    chunks: Sequence[Any] | None = None,
    **config_kwargs: Any,
) -> Any:
    """Run a function in parallel and return the combined output."""

    config = ParallelApplyConfig(**config_kwargs)
    return ParallelApplyEngine(config).run(
        data=data,
        func=func,
        func_args=func_args,
        func_kwargs=func_kwargs,
        chunks=chunks,
    ).output
