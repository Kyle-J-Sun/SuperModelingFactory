import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO, format="%(message)s")


def calculate_slope_sklearn(data, column):
    """
    Compute the slope of a data column using scikit-learn's LinearRegression.
    
    Based on ordinary least squares: fit the data points with a LinearRegression model
    and return the slope coefficient of the linear regression.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the data column
    
    Returns
    -------
    float
        Slope of the linear regression
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_sklearn(df, 'values')
    1.0
    """
    
    from sklearn.linear_model import LinearRegression
    
    series = data[column]
    # Make sure the data is a NumPy array
    y = np.array(series).reshape(-1, 1)
    
    # Create the x axis (index)
    x = np.arange(len(series)).reshape(-1, 1)
    
    # Create and fit the linear regression model
    model = LinearRegression()
    model.fit(x, y)
    
    # Get the slope
    slope = model.coef_[0][0]
    
    return slope


def calculate_slope_scipy(data, column):
    """
    Compute the slope of a data column using SciPy's linregress function.
    
    Based on ordinary least squares: fit the data points with scipy.stats.linregress
    and return the slope along with additional statistics.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the data column
    
    Returns
    -------
    tuple
        Tuple (slope, r_value, p_value, std_err) containing:
        - slope: slope value
        - r_value: correlation coefficient
        - p_value: p-value
        - std_err: standard error
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> slope, r, p, se = calculate_slope_scipy(df, 'values')
    >>> print(f"Slope: {slope}, correlation: {r}")
    Slope: 1.0, correlation: 1.0
    """
    
    from scipy import stats
    
    series = data[column]
    # Make sure the data is a NumPy array
    y = np.array(series)
    
    # Create the x axis (index)
    x = np.arange(len(y))
    
    # Run the linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    return slope, r_value, p_value, std_err


def calculate_slope_numpy(data, column):
    """
    Compute the slope of a data column using NumPy's polyfit function.
    
    Fit a first-degree polynomial with numpy.polyfit
    and return the slope of the linear regression.
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the data column
    
    Returns
    -------
    float
        Slope of the linear regression
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_numpy(df, 'values')
    1.0
    """
    
    import numpy as np
    
    series = data[column]
    # Make sure the data is a NumPy array
    y = np.array(series)
    
    # Create the x axis (index)
    x = np.arange(len(y))
    
    # Fit a first-degree polynomial (linear regression); returns the slope and intercept
    slope, intercept = np.polyfit(x, y, 1)
    
    return slope


def calculate_slope_manual(data, column):
    """
    Compute the slope of a data column manually with ordinary least squares.
    
    Implement the least-squares formula by hand to compute the slope of the linear regression:
    slope = Σ((x - x_mean) * (y - y_mean)) / Σ((x - x_mean)²)
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the data column
    
    Returns
    -------
    float
        Slope of the linear regression
    
    Examples
    --------
    >>> df = pd.DataFrame({'values': [1, 2, 3, 4, 5]})
    >>> calculate_slope_manual(df, 'values')
    1.0
    """
    
    series = data[column]
    
    # Make sure the data is a NumPy array
    y = np.array(series)
    
    # Create the x axis (index)
    x = np.arange(len(y))
    
    # Compute the means of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Compute the slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    
    return slope


class SlopeCalculator:
    """
    Slope calculator.
    
    Provide several methods to compute the linear-regression slope of a data column, supporting:
    - sklearn LinearRegression
    - scipy.stats.linregress
    - numpy.polyfit
    - manual least squares
    
    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing the data
    column : str
        Name of the data column
    
    Attributes
    ----------
    data : pandas.DataFrame
        Input data
    column : str
        Column name
    y : numpy.ndarray
        Converted data array
    x : numpy.ndarray
        x-axis array (index)
    
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
        Initialize the slope calculator.
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Name of the data column
        """
        self.data = data
        self.column = column
        self.series = data[column]
        self.y = np.array(self.series)
        self.x = np.arange(len(self.series))
    
    def calculate_sklearn(self):
        """
        Compute the slope using sklearn LinearRegression.
        
        Returns
        -------
        float
            Slope of the linear regression
        """
        return calculate_slope_sklearn(self.data, self.column)
    
    def calculate_scipy(self):
        """
        Compute the slope using scipy.stats.linregress.
        
        Returns
        -------
        tuple
            Tuple (slope, r_value, p_value, std_err)
        """
        return calculate_slope_scipy(self.data, self.column)
    
    def calculate_numpy(self):
        """
        Compute the slope using numpy.polyfit.
        
        Returns
        -------
        float
            Slope of the linear regression
        """
        return calculate_slope_numpy(self.data, self.column)
    
    def calculate_manual(self):
        """
        Compute the slope using manual least squares.
        
        Returns
        -------
        float
            Slope of the linear regression
        """
        return calculate_slope_manual(self.data, self.column)
    
    def calculate_all(self):
        """
        Compute the slope using all methods.
        
        Returns
        -------
        dict
            Dictionary with the results of each method
        """
        results = {}
        
        # sklearn method
        results['sklearn'] = self.calculate_sklearn()
        
        # scipy method
        scipy_result = self.calculate_scipy()
        results['scipy_slope'] = scipy_result[0]
        results['scipy_r_value'] = scipy_result[1]
        results['scipy_p_value'] = scipy_result[2]
        results['scipy_std_err'] = scipy_result[3]
        
        # numpy method
        results['numpy'] = self.calculate_numpy()
        
        # manual method
        results['manual'] = self.calculate_manual()
        
        return results
    
    @staticmethod
    def calculate(data, column, method='sklearn'):
        """
        Compute the slope using the specified method (static method).
        
        Parameters
        ----------
        data : pandas.DataFrame
            DataFrame containing the data
        column : str
            Name of the data column
        method : str, default 'sklearn'
            Computation method; one of 'sklearn', 'scipy', 'numpy', 'manual'
        
        Returns
        -------
        float or tuple
            Slope value (scipy returns a tuple; the other methods return a float)
        
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
            raise ValueError(f"Unsupported method: {method}. Choose one of: 'sklearn', 'scipy', 'numpy', 'manual'")
