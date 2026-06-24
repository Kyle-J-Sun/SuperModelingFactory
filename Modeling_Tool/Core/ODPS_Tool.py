from datetime import datetime
import logging
import os
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

class ODPSRunner(object):
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
        - 当 SQL 返回列数 > 200 时, ``_patch_wide_schema_download`` 自动 patch ODPS Tunnel,
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
                orig_build = None
                try:
                    logging.info(f'  to_pandas: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    reader = executor.open_reader()
                    orig_build = self._patch_wide_schema_download(reader)
                    if n_process > 1:
                        df = reader.to_pandas(n_process=n_process)
                    else:
                        df = reader.to_pandas()
                    if bool(csv_path):
                        logging.info(f'  to_csv: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                        df.to_csv(csv_path)
                    break
                except Exception as e:
                    logging.error(f'  download failed [{i+1}/{k}]: {e}')
                    if i == k - 1:
                        endtime = datetime.now()
                        duration = round((endtime - starttime).total_seconds(), 1)
                        raise SystemError(
                            f'  break: {endtime.strftime("%Y-%m-%d %H:%M:%S")} duration {duration}\n'
                        )
                finally:
                    if orig_build is not None:
                        from odps.tunnel.instancetunnel import InstanceDownloadSession
                        InstanceDownloadSession._build_input_stream = orig_build

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

    @staticmethod
    def _patch_wide_schema_download(reader, col_threshold=200):
        """Prevent HTTP 414 when a query result has many columns.

        ODPS Tunnel encodes column names as URL query params. With 200+ columns
        the URI exceeds the server limit. We temporarily replace the class-level
        _build_input_stream to omit the columns param, so the server returns all
        columns without URL-based filtering.

        Returns the original method so the caller can restore it in a finally block.
        Returns None when no patch is needed (schema is narrow enough).
        """
        ds = getattr(reader, '_download_session', None)
        if ds is None:
            return None
        schema = getattr(ds, 'schema', None)
        if schema is None or len(schema.simple_columns) <= col_threshold:
            return None

        from odps.tunnel.instancetunnel import InstanceDownloadSession
        orig_build = InstanceDownloadSession._build_input_stream

        def _build_no_column_filter(self, start, count, compress=False, columns=None, arrow=False, raw_size=None):
            return orig_build(self, start, count, compress=compress, columns=None, arrow=arrow, raw_size=raw_size)

        InstanceDownloadSession._build_input_stream = _build_no_column_filter
        logging.info(f'  wide schema ({len(schema.simple_columns)} cols): patched tunnel to omit column filter from URL')
        return orig_build

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

    def upload_df(self, df, table_name, table_schema=None, partition=None):
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
        """
        if table_schema is None:
            df.loc[:, "py_inserttime"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            table_schema = self.cre_table_schema(df=df, partition_name=None)

        self.o.delete_table(table_name, if_exists=True)
        t = self.o.create_table(table_name, table_schema)

        if bool(partition):
            with t.open_writer(partition=partition, create_partition=True) as writer:
                writer.write(df.values.tolist())
        else:
            with t.open_writer() as writer:
                writer.write(df.values.tolist())
        logger.info(f'<<<< 完成数据入表{table_name}: shape={df.shape} >>>>')

    def insert_df(self, df, table_name, overwrite=True, partition=None):
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
