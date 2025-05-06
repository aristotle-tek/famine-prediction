import json
import math
import tempfile
import warnings
import numpy as np
import pandas as pd
import pytest
from pathlib import Path




from src.model_utils import (
    load_config,
    energy_requirements,
    BMIDistribution,
    CalorieDistributor,
    calculate_energy_deficit,
    update_bmi,
    calculate_excess_mortality,
    BMIFactors,
    get_months_and_days
)



@ pytest.mark.parametrize("intake,bmi,expected_cat", [
    (2000, 14.0, "BMI < 15 kg/m²"),
    (2000, 16.0, "15 ≤ BMI < 18.5 kg/m²"),
    (2000, 20.0, "BMI ≥ 18.5 kg/m²"),
])
def test_calculate_energy_deficit_category(intake, bmi, expected_cat):
    # compute raw deficit and verify correct category uses its requirement
    req = energy_requirements[expected_cat]
    deficit = calculate_energy_deficit(intake, bmi, energy_requirements, grain_fraction=0.5)
    # deficit = (req - intake/0.5) / req, capped at 1
    expected = min((req - intake/0.5) / req, 1.0)
    assert pytest.approx(deficit, rel=1e-6) == expected


def test_calculate_energy_deficit_invalid():
    with pytest.raises(ValueError):
        calculate_energy_deficit(0, 20, energy_requirements)
    with pytest.raises(ValueError):
        calculate_energy_deficit(1000, 61, energy_requirements)
    with pytest.warns(None):
        calculate_energy_deficit(1000, 31, energy_requirements)


def test_update_bmi_effects_and_signs():
    # energy deficit should reduce BMI
    bmi_down = update_bmi(0.1, 22.0)
    assert bmi_down < 22.0
    # recovery (surplus) should increase BMI
    bmi_up = update_bmi(-0.1, 22.0, recovery=True)
    assert bmi_up > 22.0


def test_update_bmi_invalid_factors():
    bad = BMIFactors(factor_deficit=0, recovery_factor=-1)
    with pytest.raises(ValueError):
        update_bmi(0.1, 20.0, factors=bad)


def test_calculate_excess_mortality_formula():
    # below threshold uses formula
    val = calculate_excess_mortality(17.0)
    expected = 0.00028 * math.exp((18.5 - 17.0) ** 1.33)
    assert math.isclose(val, expected, rel_tol=1e-9)
    # capped at 1.0
    assert calculate_excess_mortality(-100) == 1.0
    # no excess if bmi >= 18.5
    assert calculate_excess_mortality(19.0) == 0.0


def test_get_months_and_days_correct_lengths():
    months = get_months_and_days(2024, 1, 12)
    # Jan has 31, Feb 29 (2024 leap), etc.
    assert months[0][1] == 31
    assert months[1][1] == 29
    assert months[-1][1] == 31
    assert len(months) == 12

@ pytest.mark.parametrize("method,params,check", [
    ('linear', {'top_bmi': 30}, lambda arr: np.allclose(arr[0], 30 - 12)),
    ('logarithmic', {'top_bmi': 30, 'bottom_bmi': 18}, lambda arr: arr.min() >= 18),
])
def test_bmi_distribution_range(method, params, check):
    bd = BMIDistribution(method=method, **params)
    arr = np.array(bd.get_bmi_distribution())
    assert len(arr) == 100
    assert check(arr)


def test_bmi_distribution_reference_file(tmp_path):
    df = pd.DataFrame({"data": ["bmi_init"]} | {str(i): [i] for i in range(1, 101)})
    file = tmp_path / "ref.xlsx"
    with pd.ExcelWriter(file) as w:
        df.to_excel(w, sheet_name="energy_intake_dist", index=False)
    arr = BMIDistribution(method='reference', data_file=str(file)).get_bmi_distribution()
    assert arr[0] == 100 and arr[-1] == 1


def test_calorie_distributor_linear_distribution_sum_and_monotonic():
    pop = np.ones(100)
    total = 100 * 2000
    cd = CalorieDistributor({'pop_per_percentile': pop, 'total_kcal_consumption': total})
    y = cd.linear_distribution(beta1=5)
    assert pytest.approx(np.sum(y * pop), rel=1e-6) == total
    # should increase or stay flat
    assert np.all(np.diff(y) >= -1e-6)


def test_calorie_distributor_piecewise_behavior():
    pop = np.ones(100)
    total = 100 * 2000
    cd = CalorieDistributor({'pop_per_percentile': pop, 'total_kcal_consumption': total})
    y = cd.piecewise_linear_distribution(beta1=2, c=50)
    assert pytest.approx(np.sum(y * pop), rel=1e-6) == total
    # before c, should be strictly increasing
    assert np.all(np.diff(y[:50]) > 0)
    # after c, flat
    assert np.allclose(y[50:], y[50])
