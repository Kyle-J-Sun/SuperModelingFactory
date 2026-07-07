from datetime import datetime
from contextlib import contextmanager
import logging
import os
import threading
logger = logging.getLogger(__name__)
import pandas as pd
from odps import ODPS, options
from odps.models import Schema, Column, Partition

# Available only in newer pandas versions. Older Airflow images should skip it.
try:
    pd.set_option('future.no_silent_downcasting', True)
except (KeyError, ValueError):
    pass
pd.options.mode.chained_assignment = None  # default='warn'

logging.basicConfig(level=logging.INFO, format="%(message)s")


def _split_odps_table_name(table_name):
    """Return (qualifier, bare table name) for ODPS table identifiers.

    MaxCompute accepts a project-qualified source in ``ALTER TABLE`` but the
    ``RENAME TO`` target must be the bare table name within the same project.
    """
    table_name = str(table_name)
    if "." not in table_name:
        return "", table_name
    qualifier, bare_name = table_name.rsplit(".", 1)
    return qualifier, bare_name


def _make_related_odps_table_name(table_name, suffix):
    qualifier, bare_name = _split_odps_table_name(table_name)
    related_bare = f"{bare_name}{suffix}"
    if qualifier:
        return f"{qualifier}.{related_bare}", related_bare
    return related_bare, related_bare


def _parse_odps_partition_spec(partition):
    if isinstance(partition, dict):
        return [(str(k).strip(), str(v).strip().strip("'\"")) for k, v in partition.items()]
    pairs = []
    for item in str(partition).split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"Invalid ODPS partition spec item {item!r}; expected key=value.")
        key, value = item.split("=", 1)
        pairs.append((key.strip(), value.strip().strip("'\"")))
    if not pairs:
        raise ValueError("partition must contain at least one key=value pair.")
    return pairs


def _format_odps_partition_spec(pairs, quoted=False):
    parts = []
    for key, value in pairs:
        safe_value = str(value).replace("'", "''")
        if quoted:
            parts.append(f"{key}='{safe_value}'")
        else:
            parts.append(f"{key}={safe_value}")
    return ",".join(parts)


def _make_related_odps_partition_spec(partition, suffix):
    pairs = _parse_odps_partition_spec(partition)
    related = list(pairs)
    key, value = related[-1]
    related[-1] = (key, f"{value}{suffix}")
    return _format_odps_partition_spec(related, quoted=False)


class ODPSRunner(object):
    _wide_schema_patch_lock = threading.RLock()
    _wide_schema_patch_ref_count = 0
    _wide_schema_orig_build = None
    _wide_schema_patch_active = False

    """ODPS执行类
    """
    def __init__(self):
        self.o = ODPS(
            os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
            os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
            os.environ.get("ODPS_PROJECT", "mex_anls"),
            endpoint=os.environ.get(
                "ODPS_ENDPOINT",
                "https://service.ap-southeast-1-vpc.maxcompute.aliyun-inc.com/api",
            ),
        )
        
        options.retry_times = 6         # 请求重试次数
        options.pool_maxsize = 200      # 连接池最大容量
        options.connect_timeout = 3600  # 连接超时
        options.read_timeout = 3600     # 读取超时
        

    def run_sql(self, sql, to_df=True, n_process=1, csv_path=None):
        """运行SQL并下载结果。

        Parameters
        ----------
        sql : str
            单个 SQL 代码。
        to_df : bool, default True
            是否把结果加载到内存中作为 ``pandas.DataFrame`` 返回。
            若 ``False``, 函数返回**空 DataFrame**, 但仍会下载数据(当 ``csv_path`` 被指定时)。
        n_process : int, default 1
            ``executor.open_reader().to_pandas`` 的并行进程数。
        csv_path : str, default None
            把结果另存为本地 CSV 的路径。**与 ``to_df`` 互相独立**:
                * 只设 ``csv_path`` → 下载 + 写 CSV, 不返回数据 (返回空 DataFrame)
                * 只设 ``to_df=True`` → 下载 + 返回 DataFrame, 不写 CSV
                * 两个都设 → 下载 + 返回 + 写 CSV
                * 都不设 → 只跑 SQL 不下载 (用于 DDL/INSERT 等)

        Returns
        -------
        pandas.DataFrame
            当 ``to_df=True`` 时返回完整数据;
            当 ``to_df=False`` 时返回空 DataFrame (用于占位).

        Notes
        -----
        - **执行**阶段(execute_sql)只跑一次, 无重试.
        - **下载**阶段(to_pandas + to_csv)最多重试 6 次, 适用于网络抖动.
        - 当 SQL 返回列数 > 200 时, 线程安全的 wide-schema patch 会自动 patch ODPS Tunnel,
          防止 HTTP 414 (URI too long).

        Examples
        --------
        >>> odps = ODPSRunner()
        >>> df = odps.run_sql("SELECT * FROM dual LIMIT 10")                # 仅 DataFrame
        >>> df = odps.run_sql("SELECT * FROM dual LIMIT 10", csv_path="x.csv")  # DataFrame + CSV
        >>> _  = odps.run_sql("SELECT * FROM dual LIMIT 10", to_df=False,  # 仅 CSV
        ...                   csv_path="x.csv")
        >>> _  = odps.run_sql("CREATE TABLE t AS SELECT 1")                # 仅执行, 不下载
        """
        # 准备SQL
        sqldesc = sql[:100]+"..." if len(sql)>100 else sql
        logging.info(f"SQL: \n{sqldesc}")

        # 运行SQL（只执行一次，不重试）
        starttime = datetime.now()
        logging.info(f'  execute_sql: {starttime.strftime("%Y-%m-%d %H:%M:%S")}')
        executor = self.o.execute_sql(sql)

        # 决定是否需要下载: 至少满足 to_df=True 或 csv_path 不为空
        should_download = bool(to_df) or bool(csv_path)
        df = pd.DataFrame()

        if should_download:
            k = 6
            for i in range(k):
                try:
                    logging.info(f'  to_pandas: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    reader = executor.open_reader()
                    with self._wide_schema_download_patch(reader):
                        if n_process > 1:
                            df = reader.to_pandas(n_process=n_process)
                        else:
                            df = reader.to_pandas()
                    if bool(csv_path):
                        logging.info(f'  to_csv: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                        df.to_csv(csv_path, index=False)
                    break
                except Exception as e:
                    logging.error(f'  download failed [{i+1}/{k}]: {e}')
                    if i == k - 1:
                        endtime = datetime.now()
                        duration = round((endtime - starttime).total_seconds(), 1)
                        raise SystemError(
                            f'  break: {endtime.strftime("%Y-%m-%d %H:%M:%S")} duration {duration}\n'
                        )

        # 当用户显式要求不要 DataFrame 时, 主动释放引用, 节省内存
        if not to_df:
            df = pd.DataFrame()

        endtime = datetime.now()
        duration = round((endtime - starttime).total_seconds(), 1)
        logging.info(f'  done: {endtime.strftime("%Y-%m-%d %H:%M:%S")} duration {duration}\n')
        return df

    def download_table(self, table_name, partition=None, n_process=1, csv_path=None):
        """读取表中数据至DataFrame 

        Parameters
        ----------
        table_name : str
            表名
        partition : dict
            分区, 例如: 'dt=2022-01-01,taino=0'
        n_process : int, default 1
            将查询数据转为pandas.DataFrame的进程数
        csv_path : str
            查询数据保存至csv文件路径

        Returns
        -------
        df: pandas.DataFrame
            SQL查询数据结果
        """
        logging.info(f"Table: \n{table_name} {partition}")
        starttime = datetime.now()
        logging.info(f'  to_pandas: {starttime.strftime("%Y-%m-%d %H:%M:%S")}')

        t = self.o.get_table(table_name)
        if bool(partition):
            reader = t.open_reader(partition=partition)
        else:
            reader = t.open_reader()
        if n_process > 1:
            df = reader.to_pandas(n_process=10)
        else:
            df = reader.to_pandas()

        if bool(csv_path):
            logging.info(f'  to_csv: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
            df.to_csv(csv_path, index=False)
        
        endtime = datetime.now()
        duration = round((endtime - starttime).total_seconds(), 3)
        logging.info(f'  done shape={df.shape}: {endtime.strftime("%Y-%m-%d %H:%M:%S")} duration {duration}\n')

        return df

    @classmethod
    @contextmanager
    def _wide_schema_download_patch(cls, reader, col_threshold=200):
        """Prevent HTTP 414 when a query result has many columns.

        ODPS Tunnel encodes column names as URL query params. With 200+ columns
        the URI exceeds the server limit. This context temporarily replaces the
        class-level _build_input_stream to omit the columns param. The patch is
        guarded by a lock and reference count, so concurrent threads do not
        restore the global method while another wide-schema download is active.
        """
        ds = getattr(reader, '_download_session', None)
        if ds is None:
            yield False
            return
        schema = getattr(ds, 'schema', None)
        if schema is None or len(schema.simple_columns) <= col_threshold:
            yield False
            return

        from odps.tunnel.instancetunnel import InstanceDownloadSession

        with cls._wide_schema_patch_lock:
            if not cls._wide_schema_patch_active:
                cls._wide_schema_orig_build = InstanceDownloadSession._build_input_stream

                def _build_no_column_filter(self, start, count, compress=False, columns=None, arrow=False, raw_size=None):
                    with cls._wide_schema_patch_lock:
                        orig_build = cls._wide_schema_orig_build
                    if orig_build is None:
                        raise RuntimeError("ODPS wide-schema download patch lost its original method.")
                    return orig_build(self, start, count, compress=compress, columns=None, arrow=arrow, raw_size=raw_size)

                InstanceDownloadSession._build_input_stream = _build_no_column_filter
                cls._wide_schema_patch_active = True
                logging.info(f'  wide schema ({len(schema.simple_columns)} cols): patched tunnel to omit column filter from URL')
            cls._wide_schema_patch_ref_count += 1

        try:
            yield True
        finally:
            with cls._wide_schema_patch_lock:
                cls._wide_schema_patch_ref_count -= 1
                if cls._wide_schema_patch_ref_count <= 0:
                    if cls._wide_schema_orig_build is not None:
                        InstanceDownloadSession._build_input_stream = cls._wide_schema_orig_build
                    cls._wide_schema_orig_build = None
                    cls._wide_schema_patch_ref_count = 0
                    cls._wide_schema_patch_active = False

    @staticmethod
    def cre_table_schema(df, partition_name=None):

        dtypes = df.dtypes
        isnum_dtypes = [pd.api.types.is_numeric_dtype(x) for x in dtypes]
        isint_dtypes = [pd.api.types.is_integer_dtype(x) for x in dtypes]

        table_columns = []
        table_partitions = []
        for i in range(len(df.columns)):
            col = df.columns[i]
            col_type = "string" if not isnum_dtypes[i] else "float"
            col_type = "bigint" if isint_dtypes[i] else col_type
            if col == partition_name:
                table_partitions.append(Partition(name=col, type=col_type))
            else:
                table_columns.append(Column(name=col, type=col_type))

        if bool(table_partitions):
            table_schema = Schema(columns=table_columns, partitions=table_partitions)
        else:
            table_schema = Schema(columns=table_columns)

        return table_schema

    def upload_df(self, df, table_name, table_schema=None, partition=None, atomic=True):
        """上传数据集至mc中创建新表

        Parameters
        ----------
        table_name: str
            表名
        table_schema: odps.models.Schema
            表Schema
        df: pandas.DataFrame
            数据集
        partition: string
            保存分区
        atomic: bool, default True
            When ``True`` (recommended, default from 0.4.2), the target table is
            replaced through a temp-table + rename swap so that a failure between
            ``delete_table`` and the completed write never leaves the caller
            with a dropped-and-empty table. When ``False``, the pre-0.4.2
            behaviour is used: the existing table is dropped first and then
            recreated, so any exception during write leaves the target table
            missing. Only set ``atomic=False`` if you deliberately want the
            legacy behaviour, e.g. for tables whose downstream readers expect
            a specific object identity.
        """
        if table_schema is None:
            df.loc[:, "py_inserttime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            table_schema = self.cre_table_schema(df=df, partition_name=None)

        if not atomic:
            # Legacy pre-0.4.2 non-atomic path — retained under an explicit
            # opt-out for callers that need the old identity semantics.
            self.o.delete_table(table_name, if_exists=True)
            t = self.o.create_table(table_name, table_schema)
            if bool(partition):
                with t.open_writer(partition=partition, create_partition=True) as writer:
                    writer.write(df.values.tolist())
            else:
                with t.open_writer() as writer:
                    writer.write(df.values.tolist())
            logger.info(f'<<<< 完成数据入表{table_name}: shape={df.shape} >>>>')
            return

        # Atomic swap path (default from 0.4.2). If any step before the final
        # swap fails, the original table is untouched.
        ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
        _, target_rename_name = _split_odps_table_name(table_name)
        tmp_name, _ = _make_related_odps_table_name(table_name, f"__tmp_{ts}")
        old_name, old_rename_name = _make_related_odps_table_name(table_name, f"__old_{ts}")

        # Step 1: write data into the temp table. If this fails, drop the
        # partial temp table and re-raise; the target table is untouched.
        self.o.delete_table(tmp_name, if_exists=True)
        tmp_table = self.o.create_table(tmp_name, table_schema)
        try:
            if bool(partition):
                with tmp_table.open_writer(partition=partition, create_partition=True) as writer:
                    writer.write(df.values.tolist())
            else:
                with tmp_table.open_writer() as writer:
                    writer.write(df.values.tolist())
        except Exception:
            try:
                self.o.delete_table(tmp_name, if_exists=True)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    f"upload_df: failed to drop temp table {tmp_name} after write error: {cleanup_exc!r}"
                )
            raise

        # Step 2: swap. If the original table exists, rename it aside first so
        # that the rename of tmp -> target is guaranteed to succeed. Then drop
        # the old copy (best-effort, warn on failure).
        target_exists = self.o.exist_table(table_name)
        if target_exists:
            try:
                self.o.run_sql(f"ALTER TABLE {table_name} RENAME TO {old_rename_name};")
            except Exception:
                # Rename of the live target failed — do not touch it. Drop the
                # tmp table so we don't leak it, then re-raise.
                try:
                    self.o.delete_table(tmp_name, if_exists=True)
                except Exception as cleanup_exc:  # noqa: BLE001
                    logger.warning(
                        f"upload_df: failed to drop temp table {tmp_name} after rename-target failure: {cleanup_exc!r}"
                    )
                raise

        try:
            self.o.run_sql(f"ALTER TABLE {tmp_name} RENAME TO {target_rename_name};")
        except Exception:
            # tmp -> target rename failed. If we already moved the original,
            # try to restore it so the caller is not left with a missing table.
            if target_exists:
                try:
                    self.o.run_sql(f"ALTER TABLE {old_name} RENAME TO {target_rename_name};")
                    logger.warning(
                        f"upload_df: swap of {tmp_name} -> {table_name} failed; restored original from {old_name}"
                    )
                except Exception as restore_exc:  # noqa: BLE001
                    logger.error(
                        f"upload_df: swap of {tmp_name} -> {table_name} failed AND restore of "
                        f"{old_name} -> {table_name} failed: {restore_exc!r}. "
                        f"Manual recovery required — original data is in {old_name}."
                    )
            try:
                self.o.delete_table(tmp_name, if_exists=True)
            except Exception:  # noqa: BLE001
                pass
            raise

        # Step 3: drop the moved-aside original — best-effort, do not fail the
        # call if this cleanup errors, since the swap already succeeded.
        if target_exists:
            try:
                self.o.delete_table(old_name, if_exists=True)
            except Exception as cleanup_exc:  # noqa: BLE001
                logger.warning(
                    f"upload_df: swap succeeded but failed to drop old copy {old_name}: {cleanup_exc!r}. "
                    f"Safe to delete manually."
                )

        logger.info(f'<<<< 完成数据入表{table_name}: shape={df.shape} >>>>')

    def _partition_exists(self, table, partition):
        if hasattr(table, "exist_partition"):
            return bool(table.exist_partition(partition))
        try:
            table.get_partition(partition)
            return True
        except Exception:
            return False

    def _rename_partition(self, table_name, source_partition, target_partition):
        src = _format_odps_partition_spec(_parse_odps_partition_spec(source_partition), quoted=True)
        dst = _format_odps_partition_spec(_parse_odps_partition_spec(target_partition), quoted=True)
        self.o.run_sql(f"ALTER TABLE {table_name} PARTITION ({src}) RENAME TO PARTITION ({dst});")

    def insert_df(self, df, table_name, overwrite=True, partition=None, atomic=True):
        """将数据集插入至mc已存在的表中.

        Parameters
        ----------
        df: pandas.DataFrame
            数据集
        table_name: str
            表名
        overwrite: Bool, default True
            是否覆盖
        partition: string, default None
            写入分区, 默认为None即无分区
        """
        t = self.o.get_table(table_name)
        if "py_inserttime" in t.schema and "py_inserttime" not in df.columns:
            df.loc[:, "py_inserttime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if bool(partition):
            if overwrite and atomic:
                ts = datetime.now().strftime('%Y%m%d%H%M%S%f')
                staging_partition = _make_related_odps_partition_spec(partition, f"__staging_{ts}")
                backup_partition = _make_related_odps_partition_spec(partition, f"__old_{ts}")
                target_exists = self._partition_exists(t, partition)

                for stale in (staging_partition, backup_partition):
                    try:
                        t.delete_partition(stale, if_exists=True)
                    except Exception as cleanup_exc:  # noqa: BLE001
                        logger.warning(
                            f"insert_df: failed to cleanup partition {stale} before atomic write: {cleanup_exc!r}"
                        )

                try:
                    with t.open_writer(partition=staging_partition, create_partition=True) as writer:
                        writer.write(df.values.tolist())
                    if target_exists:
                        self._rename_partition(table_name, partition, backup_partition)
                    self._rename_partition(table_name, staging_partition, partition)
                except Exception:
                    if target_exists:
                        try:
                            if not self._partition_exists(t, partition) and self._partition_exists(t, backup_partition):
                                self._rename_partition(table_name, backup_partition, partition)
                        except Exception as restore_exc:  # noqa: BLE001
                            logger.error(
                                f"insert_df: failed to restore original partition {partition} "
                                f"from {backup_partition}: {restore_exc!r}. Manual recovery may be required."
                            )
                    try:
                        t.delete_partition(staging_partition, if_exists=True)
                    except Exception:  # noqa: BLE001
                        pass
                    raise

                if target_exists:
                    try:
                        t.delete_partition(backup_partition, if_exists=True)
                    except Exception as cleanup_exc:  # noqa: BLE001
                        logger.warning(
                            f"insert_df: atomic swap succeeded but failed to drop backup partition "
                            f"{backup_partition}: {cleanup_exc!r}. Safe to delete manually."
                        )
                logger.info('<<<< insert_df atomic partition write done: shape={0} >>>>'.format(df.shape))
                return

            if overwrite:
                t.delete_partition(partition, if_exists=True)
            with t.open_writer(partition=partition, create_partition=True) as writer:
                writer.write(df.values.tolist())
        else:
            if overwrite:
                t.truncate()
            with t.open_writer() as writer:
                writer.write(df.values.tolist())
        logger.info('<<<< 完成数据入表: shape={0} >>>>'.format(df.shape))
