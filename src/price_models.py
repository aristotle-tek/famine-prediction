"""
Currently includes functions to follow the Baquedano
approach to price anomaly detection:
"Developing an indicator of price anomalies as
an early warning tool: A compound growth approach"
Felix G. Baquedano. FAO, Rome, 2015.
"""

import numpy as np
import pandas as pd



def compute_cgr(series, window):
    r"""
    Compute the compound growth rate (CGR) over a rolling window.

    Following Baquedano (2015):
    "The CGR is the growth in any random variable
    from a time period $t_0$ to $t_n$, 
    raised to the power of one over the length of
    the period of time being considered, as highlighted in Equation 1:
    
    CGR_t = (P_tn / P_t0) ** (1 / (t_n - t_0)) - 1
           = (P_last / P_first) ** (1 / (window - 1)) - 1

    Parameters:
      series (pd.Series): Time series of prices.
      window (int): Number of periods (months) to consider.
      
    Returns:
      pd.Series: The rolling CGR with NaN for periods with insufficient data.
    """
    def _cgr(x):
        periods = len(x) - 1          # elapsed months
        if periods <= 0 or x[0] == 0:
            return np.nan
        return (x[-1] / x[0]) ** (1 / periods) - 1

    return series.rolling(window).apply(_cgr, raw=True)



def compute_volatility(series, window):
    r"""
    Compute volatility as the rolling standard
    deviation of log differences.

    Baquedano (2015): "We define volatility as the
    standard deviation of log differences."
    
    Parameters:
      series (pd.Series): Time series of prices.
      window (int): Rolling window size (in months).
      
    Returns:
      pd.Series: The rolling volatility.
    """
    log_series = np.log(series)
    log_diff = log_series.diff()
    return log_diff.rolling(window - 1).std()


def adjust_cgr_for_volatility(cgr_series, vol_series, *, clip=False):
    r"""
    Deflate the CGR by (1 - σ).

    Parameters
    ----------
    cgr_series : pd.Series
    vol_series : pd.Series   # rolling std of log-diffs, ≥0
    clip       : bool        # if True, cap σ at 1 so (1-σ) never flips sign
    """
    if (vol_series < 0).any():
        raise ValueError("volatility (std) can’t be negative")

    if clip:
        vol_series = vol_series.clip(upper=1.0)

    return cgr_series * (1 - vol_series)


def weighted_mean(values, weights):
    r"""
    Compute the weighted mean of values using the given weights.

    Following Baquedano (2015) Equation (4):

    \overline{vCGR}_{Wt} = \frac{1}{\sum_{y=1}^\gamma w_y}
      \sum_{y=1}^\gamma w_y \, vCGR_{yt}
    
    Parameters:
      values (np.array): Array of values.
      weights (np.array): Array of weights.
      
    Returns:
      float: The weighted mean.
    """
    return np.sum(values * weights) / np.sum(weights)



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
    if pd.isna(score):
        return "Insufficient data"
    if score >= 1:
        return "Price Alert"
    if score >= 0.5:  # This implies score < 1 due to the check above
        return "Price Watch"
    return "Normal"



def compute_gamma(vcqgr_series, vcagr_series):
    r"""
    Compute the weight γ, determining the relative importance of the
    quarterly and annual signals via PCA on their covariance matrix.

      1. Drop any NaNs and align the two series.
      2. Build the 2×2 population covariance matrix.
      3. Compute its two (real, ≥0) eigenvalues λ₁, λ₂.
      4. Return γ = λ₁ / (λ₁ + λ₂), i.e. the proportion of total variance
         explained by the first principal component.
      5. If λ₁+λ₂ == 0 (no variability), return 0.5.

    Parameters
    ----------
    vcqgr_series : pd.Series or np.ndarray
        Volatility‐adjusted quarterly CGR.
    vcagr_series : pd.Series or np.ndarray
        Volatility‐adjusted annual CGR.

    Returns
    -------
    float
        The computed γ ∈ [0, 1].
    """

    # can drop NaNs and align indexes easier with series
    s_q = pd.Series(vcqgr_series).dropna()
    s_a = pd.Series(vcagr_series).dropna()

    # Align on their common index; if they were arrays, this just truncates to min length
    df = pd.concat([s_q, s_a], axis=1, join="inner")
    if df.shape[0] < 2:
        # too few points to estimate a covariance → equal weight
        return 0.5

    # extract as 2×N array (rows = variables, cols = observations)
    data = df.values.T

    # calc 2×2 population covariance matrix (bias=True → divide by N)
    cov = np.cov(data, bias=True)

    # for a symm matrix we can use eigvalsh (returns ascending)
    eigs = np.linalg.eigvalsh(cov)

    # order descending and sum
    eigs = np.sort(eigs)[::-1]
    total = eigs.sum()

    if total <= 0:
        return 0.5

    gamma = eigs[0] / total
    return float(gamma)


def weighted_std(values, weights):
    """
    Compute the weighted standard deviation of values
    as in Baquedano (2015) Equation (5).

    Equation (5):
    σ̂_{vCGR,Wt} = sqrt(
        (Σ wᵧ [vCGRᵧt −  vCGR̅_{Wt}]²) / 
        (Σ wᵧ × ((γ−1)/γ))
    )

    where γ is the number of observations.
    """
    mean_val = weighted_mean(values, weights)
    gamma = len(values)  # number of data points

    if gamma <= 1: # Std dev not defined for 0 or 1 point, or would be 0.
        return np.nan # Return NaN to indicate it cannot be reliably calculated.

    numerator = np.sum(weights * (values - mean_val)**2)

    # Denominator factor related to effective sample size and bias correction
    # Ensure sum_weights is not zero if gamma > 1 (should not happen with positive weights)
    sum_weights = np.sum(weights)
    if sum_weights == 0:
        return np.nan # Avoid division by zero if somehow sum of weights is zero

    denominator_factor = (gamma - 1) / gamma
    denominator = sum_weights * denominator_factor

    if denominator == 0: # Could happen if gamma=1 (already handled) or sum_weights is zero.
        return np.nan

    return np.sqrt(numerator / denominator)


def compute_anomaly_score(vcgr_value, weighted_mean_for_month, weighted_std_for_month):
    r"""
    Compute the anomaly score (X_IPA_t^Z) as per Baquedano (2015) Equation (6).

    Equation (6):
    X_IPA_t^Z = (vCGR_t - \overline{vCGR}_{Wt}) / \hat{\sigma}_{vCGR,Wt}

    Parameters:
    vcgr_value (float):
        The volatility adjusted CGR for the current month t.
    weighted_mean_for_month (float):
        The historical weighted average of vCGR for month t across years.
    weighted_std_for_month (float):
        The historical weighted standard deviation of vCGR for month t across years.

    Returns:
    float
        The anomaly score. Returns np.nan if standard deviation is zero/NaN,
        or if any input is NaN.
    """
    if pd.isna(weighted_std_for_month) or weighted_std_for_month == 0:
        return np.nan
    if pd.isna(vcgr_value) or pd.isna(weighted_mean_for_month):
        return np.nan
    return (vcgr_value - weighted_mean_for_month) / weighted_std_for_month



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
