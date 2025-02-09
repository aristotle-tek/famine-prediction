""" Utilities to test for stationarity of a series. """

from statsmodels.tsa.stattools import adfuller, kpss


IMPORT_WARNING = None
PP_IMPORT_WARNING = None

try:
    from arch.unitroot import DFGLS
except ImportError as exc:
    DFGLS = None
    IMPORT_WARNING = exc  # Store orig exception for later reference


try:
    from arch.unitroot import PhillipsPerron
except ImportError as exc:
    PhillipsPerron = None
    PP_IMPORT_WARNING = exc

def test_adf(series, regression='ct', autolag='AIC'):
    """
    Perform the Augmented Dickey–Fuller (ADF) test for a unit root.

    This test evaluates the null hypothesis that the series has a unit root (i.e. is non-stationary).
    In the context of price anomaly detection (e.g. as assumed in Baquedano's approach), a rejection
    of the null (with p-value below a chosen significance level) is consistent with trend stationarity.

    Parameters:
        series (pd.Series): The time series data.
        regression (str): The regression component to include; 'ct' (constant and trend) is recommended
                          for testing trend stationarity. Default is 'ct'.
        autolag (str): The method for automatically selecting the number of lags. Default is 'AIC'.

    Returns:
        dict: A dictionary with the test statistic, p-value, number of lags used,
              number of observations, critical values, and (if available) the information criterion.
    """
    series_clean = series.dropna()
    if series_clean.empty:
        raise ValueError("Input time series is empty after dropping NaN values.")
    
    result = adfuller(series_clean, regression=regression, autolag=autolag)
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Number of Observations': result[3],
        'Critical Values': result[4],
        'IC Best': result[5] if len(result) > 5 else None
    }


def test_pp(series, trend='ct', **kwargs):
    """
    Perform the Phillips–Perron (PP) test for a unit root.

    This test uses the null hypothesis that the series is non-stationary.
    In contexts such as Baquedano et al.'s approach to price anomaly detection,
    the PP test (which has been shown in some cases to have higher power than the ADF test)
    provides a valuable complement to other unit root tests.

    Parameters:
        series (pd.Series): The time series data.
        trend (str): Trend specification; 'ct' (constant and trend) is recommended for testing
                     trend stationarity. Other options (like 'c') can be used for level stationarity.
        **kwargs: Additional keyword arguments passed to the underlying PP test function from the arch package.

    Returns:
        dict: A dictionary with the test statistic, p-value, and critical values.
    
    Raises:
        ImportError: If the 'arch' package is not installed.
    """
    if PhillipsPerron is None:
        raise ImportError("Phillips-Perron test requires the 'arch' package. Install it via pip.") from PP_IMPORT_WARNING

    series_clean = series.dropna()
    if series_clean.empty:
        raise ValueError("Input time series is empty after dropping NaN values.")
    pp_test = PhillipsPerron(series_clean, trend=trend, **kwargs)
    return {
        'Test Statistic': pp_test.stat,
        'p-value': pp_test.pvalue,
        'Critical Values': pp_test.critical_values
    }


def test_dfgls(series, trend='ct', **kwargs):
    """
    Perform the Dickey–Fuller GLS (DF–GLS) test for a unit root.

    This test is a modified version of the ADF test that applies GLS detrending and is shown to have
    higher power than the standard ADF test. In Baquedano et al.'s approach to price anomaly detection,
    the DF–GLS (along with the PP test) was used to supplement the traditional ADF test when assessing
    trend stationarity of price series.

    Parameters:
        series (pd.Series): The time series data.
        trend (str): Specifies the type of detrending; 'ct' for constant and trend (default) is recommended.
        **kwargs: Additional keyword arguments passed to DFGLS.

    Returns:
        dict: A dictionary containing the test statistic, p-value, and critical values.

    Raises:
        ImportError: If the 'arch' package is not installed.
    """
    if DFGLS is None:
        raise ImportError("DFGLS test requires the 'arch' package. Please install it via pip.") from IMPORT_WARNING
    series_clean = series.dropna()
    if series_clean.empty:
        raise ValueError("Input time series is empty after dropping NaN values.")
    test = DFGLS(series_clean, trend=trend, **kwargs)
    return {
        'Test Statistic': test.stat,
        'p-value': test.pvalue,
        'Critical Values': test.critical_values
    }


def test_kpss(series, regression='ct', **kwargs):
    """
    Perform the Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test for stationarity.

    Unlike the ADF, PP, and DF–GLS tests (which use a null of non-stationarity), the KPSS test uses the null
    hypothesis that the series is stationary. When using regression='ct', the test evaluates trend stationarity.
    This test serves as a complementary check to the other tests.

    Parameters:
        series (pd.Series): The time series data.
        regression (str): The regression component ('c' for level stationarity or 'ct' for trend stationarity).
                          Default is 'ct'.
        **kwargs: Additional keyword arguments passed to the KPSS function.

    Returns:
        dict: A dictionary containing the test statistic, p-value, number of lags used, and critical values.
    """
    series_clean = series.dropna()
    if series_clean.empty:
        raise ValueError("Input time series is empty after dropping NaN values.")
    result = kpss(series_clean, regression=regression, **kwargs)
    return {
        'Test Statistic': result[0],
        'p-value': result[1],
        'Lags Used': result[2],
        'Critical Values': result[3]
    }