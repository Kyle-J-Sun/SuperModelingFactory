import logging
import numpy as np
import pandas as pd
# Available only in newer pandas versions. Older Airflow images should skip it.
try:
    pd.set_option('future.no_silent_downcasting', True)
except (KeyError, ValueError):
    pass
pd.options.mode.chained_assignment = None  # default='warn'
logging.basicConfig(level=logging.INFO, format="%(message)s")


def calculate_slope_sklearn(data, column):
    """
    使用SKlearn的LinearRegression计算数据列的斜率。
    
    基于最小二乘法，通过LinearRegression模型拟合数据点，
    返回线性回归的斜率系数。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        数据列名
    
    Returns
    -------
    float
        线性回归的斜率值
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_sklearn(df, 'values')
    1.0
    """
    
    from sklearn.linear_model import LinearRegression
    
    series = data[column]
    # 确保数据是NumPy数组格式
    y = np.array(series).reshape(-1, 1)
    
    # 创建x轴（索引）
    x = np.arange(len(series)).reshape(-1, 1)
    
    # 创建并拟合线性回归模型
    model = LinearRegression()
    model.fit(x, y)
    
    # 获取斜率
    slope = model.coef_[0][0]
    
    return slope


def calculate_slope_scipy(data, column):
    """
    使用SciPy的linregress函数计算数据列的斜率。
    
    基于最小二乘法，通过scipy.stats.linregress函数拟合数据点，
    返回斜率及更多统计信息。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        数据列名
    
    Returns
    -------
    tuple
        (slope, r_value, p_value, std_err) 元组，包含：
        - slope: 斜率值
        - r_value: 相关系数
        - p_value: p值
        - std_err: 标准误差
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> slope, r, p, se = calculate_slope_scipy(df, 'values')
    >>> print(f"斜率: {slope}, 相关系数: {r}")
    斜率: 1.0, 相关系数: 1.0
    """
    
    from scipy import stats
    
    series = data[column]
    # 确保数据是NumPy数组格式
    y = np.array(series)
    
    # 创建x轴（索引）
    x = np.arange(len(y))
    
    # 执行线性回归
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return slope, r_value, p_value, std_err


def calculate_slope_numpy(data, column):
    """
    使用NumPy的polyfit函数计算数据列的斜率。
    
    使用numpy.polyfit函数进行一阶多项式拟合，
    返回线性回归的斜率。
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        数据列名
    
    Returns
    -------
    float
        线性回归的斜率值
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_numpy(df, 'values')
    1.0
    """
    
    import numpy as np
    
    series = data[column]
    # 确保数据是NumPy数组格式
    y = np.array(series)
    
    # 创建x轴（索引）
    x = np.arange(len(y))
    
    # 使用一次多项式拟合（线性回归），返回斜率和截距
    slope, intercept = np.polyfit(x, y, 1)
    
    return slope


def calculate_slope_manual(data, column):
    """
    手动使用最小二乘法计算数据列的斜率。
    
    通过手动实现最小二乘法公式，计算线性回归的斜率：
    slope = Σ((x - x_mean) * (y - y_mean)) / Σ((x - x_mean)²)
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        数据列名
    
    Returns
    -------
    float
        线性回归的斜率值
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_manual(df, 'values')
    1.0
    """
    
    series = data[column]
    
    # 确保数据是NumPy数组格式
    y = np.array(series)
    
    # 创建x轴（索引）
    x = np.arange(len(y))
    
    # 计算x和y的平均值
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # 计算斜率和截距
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    
    return slope


class SlopeCalculator:
    """
    斜率计算器。
    
    提供多种方法计算数据列的线性回归斜率，支持：
    - sklearn LinearRegression
    - scipy.stats.linregress
    - numpy.polyfit
    - 手动最小二乘法
    
    Parameters
    ----------
    data : pandas.DataFrame
        包含数据的DataFrame
    column : str
        数据列名
    
    Attributes
    ----------
    data : pandas.DataFrame
        输入数据
    column : str
        列名
    y : numpy.ndarray
        转换后的数据数组
    x : numpy.ndarray
        x轴数组（索引）
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calc = SlopeCalculator(df, 'values')
    >>> calc.calculate_sklearn()
    1.0
    >>> calc.calculate_scipy()
    (1.0, 1.0, 9.999999999999996e-08, 0.0)
    """
    
    def __init__(self, data, column):
        """
        初始化斜率计算器。
        
        Parameters
        ----------
        data : pandas.DataFrame
            包含数据的DataFrame
        column : str
            数据列名
        """
        self.data = data
        self.column = column
        self.series = data[column]
        self.y = np.array(self.series)
        self.x = np.arange(len(self.series))
    
    def calculate_sklearn(self):
        """
        使用sklearn LinearRegression计算斜率。
        
        Returns
        -------
        float
            线性回归的斜率值
        """
        return calculate_slope_sklearn(self.data, self.column)
    
    def calculate_scipy(self):
        """
        使用scipy.stats.linregress计算斜率。
        
        Returns
        -------
        tuple
            (slope, r_value, p_value, std_err) 元组
        """
        return calculate_slope_scipy(self.data, self.column)
    
    def calculate_numpy(self):
        """
        使用numpy.polyfit计算斜率。
        
        Returns
        -------
        float
            线性回归的斜率值
        """
        return calculate_slope_numpy(self.data, self.column)
    
    def calculate_manual(self):
        """
        使用手动最小二乘法计算斜率。
        
        Returns
        -------
        float
            线性回归的斜率值
        """
        return calculate_slope_manual(self.data, self.column)
    
    def calculate_all(self):
        """
        使用所有方法计算斜率。
        
        Returns
        -------
        dict
            包含各种方法计算结果的字典
        """
        results = {}
        
        # sklearn方法
        results['sklearn'] = self.calculate_sklearn()
        
        # scipy方法
        scipy_result = self.calculate_scipy()
        results['scipy_slope'] = scipy_result[0]
        results['scipy_r_value'] = scipy_result[1]
        results['scipy_p_value'] = scipy_result[2]
        results['scipy_std_err'] = scipy_result[3]
        
        # numpy方法
        results['numpy'] = self.calculate_numpy()
        
        # 手动方法
        results['manual'] = self.calculate_manual()
        
        return results
    
    @staticmethod
    def calculate(data, column, method='sklearn'):
        """
        静态方法：使用指定方法计算斜率。
        
        Parameters
        ----------
        data : pandas.DataFrame
            包含数据的DataFrame
        column : str
            数据列名
        method : str, default 'sklearn'
            计算方法，候选值：'sklearn', 'scipy', 'numpy', 'manual'
        
        Returns
        -------
        float or tuple
            斜率值（scipy返回元组，其他返回浮点数）
        
        Examples
        --------
        >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
        >>> SlopeCalculator.calculate(df, 'values', method='numpy')
        1.0
        """
        calc = SlopeCalculator(data, column)
        
        if method == 'sklearn':
            return calc.calculate_sklearn()
        elif method == 'scipy':
            return calc.calculate_scipy()
        elif method == 'numpy':
            return calc.calculate_numpy()
        elif method == 'manual':
            return calc.calculate_manual()
        else:
            raise ValueError(f"不支持的方法: {method}。请选择: 'sklearn', 'scipy', 'numpy', 'manual'")
