#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utilities for the Resource dependency model."""

import math
import warnings
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from dataclasses import dataclass


def load_config(file_path):
    """
    Load a configuration file in JSON format.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
    return config


# Energy requirements based on BMI categories
energy_requirements = {
    "BMI < 15 kg/m²": 1900,
    "15 ≤ BMI < 18.5 kg/m²": 2100,
    "BMI ≥ 18.5 kg/m²": 2200
}


def calculate_energy_deficit(intake_kcal, bmi, energy_req_dict, grain_fraction=0.7):
    """
    Calculate the energy deficit (positive) or surplus (negative) as a fraction of the energy requirement.

    Args:
        intake_kcal (float): The cereal intake in kcal (must be positive).
        bmi (float): The current BMI of the individual.
        energy_req_dict (dict): Dictionary of energy requirements keyed by BMI category.
        grain_fraction (float): The fraction of calories from grains (0 < value ≤ 1; default 0.7).

    Returns:
        float: The energy deficit (or surplus, capped at 100%).
    """
    # Input validation
    if not isinstance(intake_kcal, (int, float, np.int64, np.float64)) or intake_kcal <= 0:
        raise ValueError("Cereal intake must be a positive number.")

    if bmi > 60:
        raise ValueError("BMI should not exceed 60.")

    if bmi < 15 or bmi > 30:
        warnings.warn("Warning: extreme values of BMI (<15 or >30).")

    if not isinstance(grain_fraction, (int, float)) or not 0 < grain_fraction <= 1:
        raise ValueError("Percent grain must be a number between 0 and 1.")

    # Validate energy requirements dictionary
    required_bmi_categories = [
        "BMI < 15 kg/m²",
        "15 ≤ BMI < 18.5 kg/m²",
        "BMI ≥ 18.5 kg/m²"
    ]
    if not all(category in energy_req_dict for category in required_bmi_categories):
        raise ValueError("The energy requirements dictionary must contain all required BMI categories.")

    # Determine BMI category
    if bmi < 15:
        bmi_category = "BMI < 15 kg/m²"
    elif 15 <= bmi < 18.5:
        bmi_category = "15 ≤ BMI < 18.5 kg/m²"
    else:
        bmi_category = "BMI ≥ 18.5 kg/m²"

    energy_requirement = energy_req_dict[bmi_category]
    energy_intake = intake_kcal / grain_fraction
    deficit_or_surplus = min((energy_requirement - energy_intake) / energy_requirement, 1.0)

    return deficit_or_surplus


@dataclass(frozen=True)
class BMIFactors:
    factor_deficit: float = 0.091
    factor_adj: float = 3.41316
    recovery_factor: float = 3.52


def update_bmi(energy_deficit, previous_bmi, recovery=False, factors: BMIFactors = BMIFactors()):
    """
    Calculate the updated BMI based on energy deficit/surplus and previous BMI.

    Args:
        energy_deficit (float): Energy deficit (positive for deficit, negative for surplus) as a fraction.
        previous_bmi (float): Previous month's BMI.
        recovery (bool): Flag indicating recovery (surplus) scenario.
        factors (BMIFactors): Coefficients used for BMI adjustment.

    Returns:
        float: Updated BMI.
    """
    if not isinstance(energy_deficit, (int, float)):
        raise ValueError("Deficit must be a number.")

    if previous_bmi > 60:
        raise ValueError("The previous BMI should not exceed 60.")

    if factors.factor_deficit <= 0 or factors.recovery_factor <= 0:
        raise ValueError("Factors should be positive numbers.")

    if recovery:
        delta_bmi = (factors.recovery_factor * energy_deficit) - (
            factors.factor_deficit * (22 - previous_bmi)
        )
    else:
        delta_bmi = (factors.factor_adj * energy_deficit) - (
            factors.factor_deficit * (22 - previous_bmi)
        )

    return previous_bmi - delta_bmi


def calculate_excess_mortality(bmi_prev):
    """
    Calculate the excess mortality based on the previous month's BMI.
    
    The formula used is:
    $$0.00028 \\times e((18.5 - \\text{bmi_prev})^{1.33})$$
    for bmi_prev below 18.5, capped at 1.0.
    
    Args:
        bmi_prev (float): BMI value from the previous month.

    Returns:
        float: Excess mortality value.
    """
    if bmi_prev >= 18.5:
        return 0.0
    excess_mortality = 0.00028 * math.exp((18.5 - bmi_prev) ** 1.33)
    return min(excess_mortality, 1.0)


# class CalorieDistributor:
#     """
#     Class for distributing caloric intake across percentiles based on population data and constraints.
#     """
#     def __init__(self, pop_per_percentile, total_kcal_consumption, kcal_min=700, kcal_max=1540, epsilon=0.001):
#         self.pop_per_percentile = np.array(pop_per_percentile)
#         self.total_kcal_consumption = total_kcal_consumption
#         self.kcal_min = kcal_min
#         self.kcal_max = kcal_max
#         self.epsilon = epsilon
#         self.percentiles = np.arange(1, 101)

class CalorieDistributor:
    """
    Class for distributing caloric intake across percentiles based on population data and constraints.
    """
    def __init__(self, calorie_config):
        """
        Expects a calorie_config dictionary with keys:
            pop_per_percentile (array-like),
            total_kcal_consumption (float),
            kcal_min (float, optional),
            kcal_max (float, optional),
            epsilon (float, optional).
        """
        self.pop_per_percentile = np.array(calorie_config['pop_per_percentile'])
        self.total_kcal_consumption = calorie_config['total_kcal_consumption']
        self.kcal_min = calorie_config.get('kcal_min', 700)
        self.kcal_max = calorie_config.get('kcal_max', 1540)
        self.epsilon = calorie_config.get('epsilon', 0.001)
        self.percentiles = np.arange(1, 101)

    def linear_distribution(self, beta1):
        """
        Distribute calories linearly across percentiles using slope beta1.
        Adjusts distribution by capping at kcal_min or kcal_max.
        """
        def calculate_distribution(max_kcal):
            pop = self.pop_per_percentile
            x = self.percentiles

            beta0_min = self.kcal_min - beta1 * 100
            beta0_max = max_kcal - beta1

            def total_kcal(beta0):
                y = beta0 + beta1 * x
                y = np.clip(y, self.kcal_min, max_kcal)
                return np.sum(y * pop)

            total_min = total_kcal(beta0_min)
            total_max = total_kcal(beta0_max)

            if total_min - self.total_kcal_consumption > self.epsilon * self.total_kcal_consumption:
                def total_kcal_diff_lower(c):
                    idx_lower = x <= c
                    idx_upper = x > c
                    s2 = np.sum(pop[idx_upper])
                    s2_x = np.sum(pop[idx_upper] * x[idx_upper])
                    beta0 = (self.total_kcal_consumption - self.kcal_min * np.sum(pop[idx_lower])
                             - beta1 * s2_x) / s2
                    y_upper = beta0 + beta1 * x[idx_upper]
                    y_upper = np.clip(y_upper, self.kcal_min, max_kcal)
                    total = self.kcal_min * np.sum(pop[idx_lower]) + np.sum(y_upper * pop[idx_upper])
                    return total - self.total_kcal_consumption

                c_lower, c_upper = 1, 99
                f_lower, f_upper = total_kcal_diff_lower(c_lower), total_kcal_diff_lower(c_upper)
                if f_lower * f_upper > 0:
                    return None
                c_opt = brentq(total_kcal_diff_lower, c_lower, c_upper, xtol=0.01)
                idx_lower = x <= c_opt
                idx_upper = x > c_opt
                s2 = np.sum(pop[idx_upper])
                s2_x = np.sum(pop[idx_upper] * x[idx_upper])
                beta0 = (self.total_kcal_consumption - self.kcal_min * np.sum(pop[idx_lower])
                         - beta1 * s2_x) / s2
                y = np.zeros_like(x, dtype=float)
                y[idx_lower] = self.kcal_min
                y[idx_upper] = beta0 + beta1 * x[idx_upper]
                y = np.clip(y, self.kcal_min, max_kcal)
            elif self.total_kcal_consumption - total_max > self.epsilon * self.total_kcal_consumption:
                def total_kcal_diff_upper(c):
                    idx_lower = x <= c
                    idx_upper = x > c
                    s1 = np.sum(pop[idx_lower])
                    s1_x = np.sum(pop[idx_lower] * x[idx_lower])
                    beta0 = (self.total_kcal_consumption - max_kcal * np.sum(pop[idx_upper])
                             - beta1 * s1_x) / s1
                    y_lower = beta0 + beta1 * x[idx_lower]
                    y_lower = np.clip(y_lower, self.kcal_min, max_kcal)
                    total = np.sum(y_lower * pop[idx_lower]) + max_kcal * np.sum(pop[idx_upper])
                    return total - self.total_kcal_consumption

                c_lower, c_upper = 1, 99
                f_lower, f_upper = total_kcal_diff_upper(c_lower), total_kcal_diff_upper(c_upper)
                if f_lower * f_upper > 0:
                    return None
                c_opt = brentq(total_kcal_diff_upper, c_lower, c_upper, xtol=0.01)
                idx_lower = x <= c_opt
                idx_upper = x > c_opt
                s1 = np.sum(pop[idx_lower])
                s1_x = np.sum(pop[idx_lower] * x[idx_lower])
                beta0 = (self.total_kcal_consumption - max_kcal * np.sum(pop[idx_upper])
                         - beta1 * s1_x) / s1
                y = np.zeros_like(x, dtype=float)
                y[idx_lower] = beta0 + beta1 * x[idx_lower]
                y[idx_upper] = max_kcal
                y = np.clip(y, self.kcal_min, max_kcal)
            else:
                def total_kcal_diff(beta0):
                    y = beta0 + beta1 * x
                    y = np.clip(y, self.kcal_min, max_kcal)
                    return np.sum(y * pop) - self.total_kcal_consumption

                f_min = total_kcal_diff(beta0_min)
                f_max = total_kcal_diff(beta0_max)
                if f_min * f_max > 0:
                    return None
                beta0 = brentq(total_kcal_diff, beta0_min, beta0_max, xtol=self.epsilon)
                y = beta0 + beta1 * x
                y = np.clip(y, self.kcal_min, max_kcal)
            return y

        # First attempt with self.kcal_max; if needed, adjust max kcal.
        y = calculate_distribution(self.kcal_max)
        if y is None or np.sum(y * self.pop_per_percentile) < self.total_kcal_consumption:
            warnings.warn("Raising kcal_max to 1820 due to excess calories.")
            y = calculate_distribution(1820)

        if y is None:
            warnings.warn("No valid distribution found; using fallback uniform distribution.")
            y = np.full(100, self.total_kcal_consumption / np.sum(self.pop_per_percentile))

        total = np.sum(y * self.pop_per_percentile)
        if abs(total - self.total_kcal_consumption) > self.epsilon * self.total_kcal_consumption:
            warnings.warn("Total kcal consumption does not match within epsilon after adjustments.")
        return y

    def piecewise_linear_distribution(self, beta1, c):
        """
        Distribute calories using a piecewise linear function: increasing up to
        percentile c, then flat.
        """
        pop = self.pop_per_percentile
        x = self.percentiles

        idx_lower = x <= c
        idx_upper = x > c

        total_pop = np.sum(pop)
        weighted_offset = np.sum((c - x[idx_lower]) * pop[idx_lower])
        y_c = (self.total_kcal_consumption + beta1 * weighted_offset) / total_pop

        if y_c > self.kcal_max:
            y_c = self.kcal_max
            warnings.warn("y_c adjusted to kcal_max to satisfy kcal_max constraint.")
        elif y_c < self.kcal_min:
            y_c = self.kcal_min
            warnings.warn("y_c adjusted to kcal_min to satisfy kcal_min constraint.")

        y_min = y_c - beta1 * (c - np.min(x[idx_lower]))
        if y_min < self.kcal_min:
            beta1 = (y_c - self.kcal_min) / (c - np.min(x[idx_lower]))
            warnings.warn("beta1 adjusted to satisfy kcal_min constraint at the lowest percentile.")

        weighted_offset = np.sum((c - x[idx_lower]) * pop[idx_lower])
        y_c = (self.total_kcal_consumption + beta1 * weighted_offset) / total_pop

        y = np.zeros_like(x, dtype=float)
        y[idx_lower] = y_c - beta1 * (c - x[idx_lower])
        y[idx_upper] = y_c
        y = np.clip(y, self.kcal_min, self.kcal_max)

        total = np.sum(y * pop)
        if abs(total - self.total_kcal_consumption) > self.epsilon * self.total_kcal_consumption:
            warnings.warn("Total kcal consumption does not match within epsilon after adjustments.")
        return y


class BMIDistribution:
    """
    Generate the initial BMI distribution using one of several methods.
    """
    def __init__(self, method='linear', top_bmi=30, bottom_bmi=18, data_file=None):
        self.percentiles = np.arange(1, 101)
        self.bmi_init = None

        if method == 'linear':
            self.bmi_init = self._assign_initial_bmi_linear(top_bmi)
        elif method == 'logarithmic':
            self.bmi_init = self._assign_initial_bmi_logarithmic(top_bmi, bottom_bmi)
        elif method == 'reference':
            if data_file is None:
                raise ValueError("A data file must be provided for the 'reference' method.")
            self.bmi_init = self._assign_initial_bmi_reference(data_file)
        else:
            raise ValueError("Invalid method. Choose 'linear', 'logarithmic', or 'reference'.")

    def _assign_initial_bmi_linear(self, top_bmi):
        return [top_bmi - ((percentile - 1) * (12 / 99))
                for percentile in self.percentiles[::-1]]

    def _assign_initial_bmi_logarithmic(self, top_bmi, bottom_bmi, alpha=0.7):
        delta_bmi = top_bmi - bottom_bmi
        max_ln = np.log(100)
        return [bottom_bmi + delta_bmi * (np.log(101 - percentile) / max_ln) ** alpha
                for percentile in self.percentiles][::-1]

    def _assign_initial_bmi_reference(self, data_file):
        dfbmi = pd.read_excel(data_file, sheet_name="energy_intake_dist")
        bmi_init = dfbmi[dfbmi.data == "bmi_init"].values
        relvalues = bmi_init[0][1:]
        return relvalues[::-1]

    def get_bmi_distribution(self):
        return self.bmi_init


def get_months_and_days(start_year, start_month, num_months):
    """
    Return a list of tuples containing a month's timestamp and the number of days in that month.
    """
    months_and_days = []
    current_date = pd.to_datetime(f'{start_year}-{start_month}-01')
    for _ in range(num_months):
        month = current_date
        days_in_month = current_date.days_in_month
        months_and_days.append((month, days_in_month))
        current_date = current_date + pd.DateOffset(months=1)
    return months_and_days


# Example usage (partial)
if __name__ == "__main__":
    # Example 1: Deficit scenario
    CEREAL_INTAKE = 1400  # kcal
    BMI_PREV = 20.0
    PERCENT_GRAIN = 0.7

    deficit = calculate_energy_deficit(CEREAL_INTAKE, BMI_PREV, energy_requirements, PERCENT_GRAIN)
    bmi_new = update_bmi(deficit, BMI_PREV)
    print(f"Deficit: {deficit:.2f}")
    print(f"Previous BMI: {BMI_PREV}")
    print(f"Updated BMI: {bmi_new:.2f}")
    mortality = calculate_excess_mortality(BMI_PREV)
    print(f"Excess mortality: {mortality:.2f}")

    # Example 2: Surplus (recovery) scenario
    CEREAL_INTAKE = 1800  # kcal
    BMI_PREV = 18.0
    PERCENT_GRAIN = 0.7

    deficit = calculate_energy_deficit(CEREAL_INTAKE, BMI_PREV, energy_requirements, PERCENT_GRAIN)
    bmi_new = update_bmi(deficit, BMI_PREV)
    print(f"\nDeficit: {deficit:.2f}")
    print(f"Previous BMI: {BMI_PREV}")
    print(f"Updated BMI: {bmi_new:.2f}")

    # Example 3: Extreme BMI value
    CEREAL_INTAKE = 1600  # kcal
    BMI_PREV = 14.0  # Extreme low BMI
    PERCENT_GRAIN = 0.7

    deficit = calculate_energy_deficit(CEREAL_INTAKE, BMI_PREV, energy_requirements, PERCENT_GRAIN)
    bmi_new = update_bmi(deficit, BMI_PREV)
    mortality = calculate_excess_mortality(BMI_PREV)
    print(f"\nDeficit: {deficit:.2f}")
    print(f"Previous BMI: {BMI_PREV}")
    print(f"Updated BMI: {bmi_new:.2f}")
    print(f"Excess mortality: {mortality:.2f}")

