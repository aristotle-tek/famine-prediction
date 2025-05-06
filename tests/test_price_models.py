""" Simple tests of price models."""
import numpy as np
import pandas as pd
from src.price_models import compute_gamma

def test_compute_gamma_few_points():
    """ just 2 points """
    assert compute_gamma(np.array([1.0]), np.array([2.0])) == 0.5

def test_compute_gamma_zero_variability():
    """ 1,1 and 2,2 """
    assert compute_gamma(np.array([1, 1]), np.array([2, 2])) == 0.5

def test_compute_gamma_perfect_correlation():
    """corr near 1 """
    arr = np.array([1, 2, 3, 4])
    gamma_identical = compute_gamma(arr, arr)
    assert abs(gamma_identical - 1.0) < 1e-8

def test_compute_gamma_with_nan_series():
    """ test missing val """
    s_q = pd.Series([np.nan, 1, 2, 3])
    s_a = pd.Series([np.nan, 2, 4, 6])
    gamma_series = compute_gamma(s_q, s_a)
    assert abs(gamma_series - 1.0) < 1e-8
