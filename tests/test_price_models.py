
import numpy as np
import pandas as pd
from price_models import compute_gamma

def test_compute_gamma():
    # too few points → equal weight
    assert compute_gamma(np.array([1.0]), np.array([2.0])) == 0.5

    # zero variability in both → total variance = 0 → equal weight
    assert compute_gamma(np.array([1, 1]), np.array([2, 2])) == 0.5

    # perfectly correlated series → first PC explains 100% of variance
    arr = np.array([1, 2, 3, 4])
    gamma_identical = compute_gamma(arr, arr)
    assert abs(gamma_identical - 1.0) < 1e-8

    # with pandas Series and NaNs (drops NaNs before computing)
    s_q = pd.Series([np.nan, 1, 2, 3])
    s_a = pd.Series([np.nan, 2, 4, 6])
    gamma_series = compute_gamma(s_q, s_a)
    assert abs(gamma_series - 1.0) < 1e-8

    print("All compute_gamma tests passed.")

if __name__ == "__main__":
    test_compute_gamma()