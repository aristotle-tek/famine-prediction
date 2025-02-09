"""
Currently includes functions to follow the Baquedano approach to price anomaly detection:
"Developing an indicator of price anomalies as an early warning tool: A compound growth approach"
Felix G. Baquedano. FAO, Rome, 2015.
"""

import numpy as np

def compute_cgr(series, window):
    r"""
    Compute the compound growth rate (CGR) over a rolling window.

    Following Baquedano (2015):
    "The CGR is the growth in any random variable from a time period $t_0$ to $t_n$, 
    raised to the power of one over the length of the period of time being considered, as highlighted in Equation 1:
    
    CGR_t = \left(\frac{P_{t_n}}{P_{t_0}}\right)^{\frac{1}{t_n-t_0}} - 1
    
    Parameters:
      series (pd.Series): Time series of prices.
      window (int): Number of periods (months) to consider.
      
    Returns:
      pd.Series: The rolling CGR with NaN for periods with insufficient data.
    """
    return series.rolling(window).apply(
        lambda x: np.nan if x[0] == 0 else (x[-1] / x[0])**(1/window) - 1, raw=True
    )

def compute_volatility(series, window):
    r"""
    Compute volatility as the rolling standard deviation of log differences.

    Baquedano (2015): "We define volatility as the standard deviation of log differences."
    
    Parameters:
      series (pd.Series): Time series of prices.
      window (int): Rolling window size (in months).
      
    Returns:
      pd.Series: The rolling volatility.
    """
    log_series = np.log(series)
    log_diff = log_series.diff()
    return log_diff.rolling(window).std()

def adjust_cgr_for_volatility(cgr_series, vol_series):
    r"""
    Adjust the compound growth rate (CGR) for volatility.

    Baquedano (2015):
    "By reducing the slope of the compound growth rate, the deviations with respect to the average of the compound growth rate 
    at month $t$, will be smaller ... we deflate the CGR by its respective measure, as shown in Equation 3:
    
    vCGR_t = CGR_t \times (1 - \sigma_{[P_{t_0}-P_{t_n}]} )
    
    Parameters:
      cgr_series (pd.Series): Rolling CGR.
      vol_series (pd.Series): Corresponding rolling volatility.
      
    Returns:
      pd.Series: The volatility-adjusted CGR (vCGR).
    """
    return cgr_series * (1 - vol_series)

def weighted_mean(values, weights):
    r"""
    Compute the weighted mean of values using the given weights.

    Following Baquedano (2015) Equation (4):

    \overline{vCGR}_{Wt} = \frac{1}{\sum_{y=1}^\gamma w_y} \sum_{y=1}^\gamma w_y \, vCGR_{yt}
    
    Parameters:
      values (np.array): Array of values.
      weights (np.array): Array of weights.
      
    Returns:
      float: The weighted mean.
    """
    return np.sum(values * weights) / np.sum(weights)

def weighted_std(values, weights):
    r"""
    Compute the weighted standard deviation.

    Baquedano (2015) Equation (5):
    \hat{\sigma}_{vCGR,Wt} = \sqrt{\frac{\sum_{y=1}^\gamma w_y (vCGR_{yt} - \overline{vCGR}_{Wt})^2}{\sum_{y=1}^\gamma w_y (\gamma-1)/\gamma}}
    
    Parameters:
      values (np.array): Array of values.
      weights (np.array): Array of weights.
      
    Returns:
      float: The weighted standard deviation.
    """
    mean_val = weighted_mean(values, weights)
    N = len(values)
    return np.sqrt(np.sum(weights * (values - mean_val)**2) / (np.sum(weights) * ((N - 1) / N)))

def compute_anomaly_score(vcgr, w_mean, w_std):
    r"""
    Compute the anomaly score for a volatility-adjusted CGR (vCGR).

    Baquedano (2015) Equation (6), the anomaly score is defined as:

    IPA_t^Z = \frac{vCGR_t - \overline{vCGR}_{Wt}}{\sigma_{vCGR,Wt}}
    
    This represents the number of weighted standard deviations that the current vCGR deviates 
    from its weighted mean.
    
    Parameters:
      vcgr (float): The current volatility-adjusted CGR.
      w_mean (float): The weighted mean of vCGR values.
      w_std (float): The weighted standard deviation of vCGR values.
      
    Returns:
      float: The anomaly score.
    """
    return (vcgr - w_mean) / w_std

def classify_anomaly(score):
    r"""
    Classify the anomaly based on the anomaly score.

    Baquedano (2015):
    
    - "Price Watch" if $0.5 \le IPA_t^Z < 1$,
    - "Price Alert" if $IPA_t^Z \ge 1$,
    - "Normal" otherwise.
    
    Parameters:
      score (float): The anomaly score.
      
    Returns:
      str: The classification ("Price Alert", "Price Watch", or "Normal").
    """
    if score >= 1:
        return "Price Alert"
    if score >= 0.5:
        return "Price Watch"
    return "Normal"

def compute_gamma(vcqgr_series, vcagr_series):
    r"""
    Compute the weight $\gamma$, determining the relative importance of the quarterly and annual signals.

    Baquedano (2015):
    
    "A critical component of the IPA_t is the value of \gamma. ... The PCA allows us to calculate the eigenvalues 
    for both of these compound growth rates. The ratio of each eigenvalue to the sum of the variances gives us the value for \gamma."
    
    Parameters:
      vcqgr_series (pd.Series or np.array): Volatility-adjusted quarterly CGR.
      vcagr_series (pd.Series or np.array): Volatility-adjusted annual CGR.
      
    Returns:
      float: The computed $\gamma$.
    """
    var_q = np.nanvar(vcqgr_series)
    var_a = np.nanvar(vcagr_series)
    if (var_q + var_a) == 0:
        return 0.5  # default to equal weighting if both variances are zero
    return var_q / (var_q + var_a)

def combine_signals(ipa_quarterly, ipa_annual, gamma):
    r"""
    Combine the quarterly and annual anomaly scores into a final indicator.

    Following Baquedano (2015) Equation (7):
    
    IPA_t = \gamma \times IPA_{quarterly} + (1-\gamma) \times IPA_{annual}
    
    Parameters:
      ipa_quarterly (float): Anomaly score from the quarterly (3-month) data.
      ipa_annual (float): Anomaly score from the annual (12-month) data.
      gamma (float): Weight determining the relative importance of the quarterly signal.
      
    Returns:
      float: The combined anomaly indicator.
    """
    return gamma * ipa_quarterly + (1 - gamma) * ipa_annual

def handle_missing_data(series, method='interpolate', **kwargs):
    """
    Fill missing values in a time series using a specified method.

    Available methods:
      - 'interpolate': Use time-based linear interpolation (default).
      - 'ffill': Forward fill.
      - 'bfill': Backward fill.
      - 'drop': (can lead to misaligned series).

    Additional keyword arguments are passed to the underlying pandas function.
    """
    if method == 'interpolate':
        # Time-based interpolation is recommended when the index is datetime.
        return series.interpolate(method='time', **kwargs)
    if method == 'ffill':
        return series.fillna(method='ffill', **kwargs)
    if method == 'bfill':
        return series.fillna(method='bfill', **kwargs)
    if method == 'drop':
        return series.dropna()
    raise ValueError("Unsupported method. Choose 'interpolate', 'ffill', 'bfill', or 'drop'.")
    

