""" Simple tests of the resource scarcity model."""
import math
import numpy as np
import pandas as pd
import pytest



from src.model_utils import (
    energy_requirements,
    BMIDistribution,
    CalorieDistributor,
    calculate_energy_deficit,
    update_bmi,
    calculate_excess_mortality,
    BMIFactors,
)


@pytest.mark.parametrize("bmi, req_key", [
    (14.9, "BMI < 15 kg/m²"),
    (15.0, "15 ≤ BMI < 18.5 kg/m²"),
    (18.49, "15 ≤ BMI < 18.5 kg/m²"),
    (18.5, "BMI ≥ 18.5 kg/m²"),
])
def test_category_selection_only(bmi, req_key):
    """compute raw deficit and verify correct category uses its requirement"""
    deficit = calculate_energy_deficit(1, bmi, energy_requirements, grain_fraction=1)
    # Check that the *same* requirement the prod code would pick is used.
    used_req = next(k for k, v in energy_requirements.items()
                    if math.isclose(deficit, (v - 1) / v, rel_tol=1e-9))
    assert used_req == req_key


def test_calculate_energy_deficit_invalid():
    """ Trivial test"""
    with pytest.raises(ValueError):
        calculate_energy_deficit(0, 20, energy_requirements)
    with pytest.raises(ValueError):
        calculate_energy_deficit(1000, 61, energy_requirements)
    with pytest.warns(UserWarning):
        calculate_energy_deficit(1000, 31, energy_requirements)


@pytest.mark.parametrize("bad_fraction", [-0.1, 0, 1.2, "foo"])
def test_bad_grain_fraction(bad_fraction):
    with pytest.raises(ValueError):
        calculate_energy_deficit(1000, 20, energy_requirements, grain_fraction=bad_fraction)


def test_update_bmi_effects_and_signs():
    """Energy deficit should reduce BMI"""
    bmi_down = update_bmi(0.1, 22.0)
    assert bmi_down < 22.0
    # recovery (surplus) should increase BMI
    bmi_up = update_bmi(-0.1, 22.0, recovery=True)
    assert bmi_up > 22.0


def test_update_bmi_invalid_factors():
    """ Direction check """
    bad = BMIFactors(factor_deficit=0, recovery_factor=-1)
    with pytest.raises(ValueError):
        update_bmi(0.1, 20.0, factors=bad)


def test_update_bmi_exact():
    """Add numerical oracle using the public BMIFactors   """ 
    factors = BMIFactors()
    deficit = 0.2
    prev = 21
    expected = prev - ((factors.factor_adj * deficit)
                       - factors.factor_deficit * (22 - prev))
    assert update_bmi(deficit, prev) == pytest.approx(expected)


def test_calculate_excess_mortality_formula():
    """below threshold uses formula"""
    val = calculate_excess_mortality(17.0)
    expected = 0.00028 * math.exp((18.5 - 17.0) ** 1.33)
    assert math.isclose(val, expected, rel_tol=1e-9)
    # capped at 1.0
    assert calculate_excess_mortality(-100) == 1.0
    # no excess if bmi >= 18.5
    assert calculate_excess_mortality(19.0) == 0.0



@ pytest.mark.parametrize("method,params,check", [
    ('linear', {'top_bmi': 30}, lambda arr: np.allclose(arr[0], 30 - 12)),
    ('logarithmic', {'top_bmi': 30, 'bottom_bmi': 18}, lambda arr: arr.min() >= 18),
])
def test_bmi_distribution_range(method, params, check):
    """ trivial - should have len 100"""
    bd = BMIDistribution(method=method, **params)
    arr = np.array(bd.get_bmi_distribution())
    assert len(arr) == 100
    assert check(arr)



def test_reference_loader_roundtrip(tmp_path):
    """Validates Excel-loading path"""
    dummy = np.arange(1, 101)
    df = pd.DataFrame({"data": ["bmi_init"], **{str(i): [v] for i, v in enumerate(dummy, 1)}})
    f = tmp_path / "ref.xlsx"
    with pd.ExcelWriter(f) as w: df.to_excel(w, sheet_name="energy_intake_dist", index=False)
    loaded = BMIDistribution(method="reference", data_file=f).get_bmi_distribution()
    assert np.array_equal(loaded[::-1], dummy)



def test_calorie_distributor_linear_distribution_sum_and_monotonic():
    """ ensure monotonically non-decreasing """
    pop = np.ones(100)
    total = 100 * 2000
    cd = CalorieDistributor({'pop_per_percentile': pop, 'total_kcal_consumption': total})
    y = cd.linear_distribution(beta1=5)
    assert pytest.approx(np.sum(y * pop), rel=1e-6) == total
    # should increase or stay flat
    assert np.all(np.diff(y) >= -1e-6)



def test_calorie_distributor_piecewise_behavior():
    """monotonically non-decreasing """
    pop = np.ones(100)
    total = 100 * 1400
    cd = CalorieDistributor({'pop_per_percentile': pop, 'total_kcal_consumption': total})
    y = cd.piecewise_linear_distribution(beta1=2, c=50)
    assert pytest.approx(np.sum(y * pop), rel=1e-6) == total
    # before c, should be strictly increasing
    assert np.all(np.diff(y[:50]) > 0)
    # after c, flat
    c = 50
    assert np.allclose(y[c:], y[c])
    assert y[:c].max() <= y[c]           # strict plateau

