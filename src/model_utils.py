#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Utilities for the Resource dependency model."""

import math
import warnings
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq


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
    Calculate the energy deficit or surplus as a fraction of the energy requirement.

    Args:
        intake_kcal (float): The cereal intake in kcal (must be positive).
        bmi (float): The current BMI of the individual.
        energy_req_dict (dict): A dictionary of energy requirements based on BMI categories.
        grain_fraction (float): The percentage of calories from grains (must be between 0 and 1, default is 0.7).

    Returns:
        float: The energy deficit (positive for deficit, negative for surplus) as a fraction of the energy requirement.
    """
   # Input validation
    if not isinstance(intake_kcal, (int, float, np.int64, np.float64)) or intake_kcal <= 0:
        raise ValueError("Cereal intake must be a positive number.")

    if not bmi <= 60:
        raise ValueError("BMI should not exceed 60.")
    #if not (10 <= bmi <= 60):
    #    raise ValueError("BMI should be in the range of 10 to 60.")

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

    # Calculate energy requirement and intake
    energy_requirement = energy_req_dict[bmi_category]
    energy_intake = intake_kcal / grain_fraction

    # Calculate energy deficit or surplus fraction
    deficit_or_surplus = min((energy_requirement - energy_intake) / energy_requirement, 1.0) # Positive for deficit, negative for surplus, max 100%

    return deficit_or_surplus



def update_bmi(energy_deficit, previous_bmi, recovery=False, factor_deficit=0.091, factor_adj=3.41316, recovery_factor=3.52):
    """
    Calculate the updated BMI based on energy deficit/surplus and previous BMI.

    Args:
        energy_deficit (float): Energy deficit (positive for deficit, negative for surplus) as a fraction of the energy requirement.
        previous_bmi (float): The BMI of the previous month.
        recovery (bool): whether in recovery
        factor_deficit (float): Coefficient for BMI adjustment during deficit (0.0914 or 0.096).
        recovery_factor (float): Coefficient for BMI adjustment during recovery

    Returns:
        float: The updated BMI for the current month.
    """
    # Input validation
    if not isinstance(energy_deficit, (int, float)):
        raise ValueError("Deficit must be a number.")

    # if not (10 <= previous_bmi <= 60):
    #     raise ValueError("The previous BMI should be in the range of 10 to 60.")
    if not previous_bmi <= 60:
        raise ValueError("The previous BMI should not exceed 60.")


    if factor_deficit <= 0 or recovery_factor <= 0:
        raise ValueError("Factors should be positive numbers.")

    # Adjust BMI based on deficit or surplus
    if recovery:
        # Recovery (surplus) case
        delta_bmi = (recovery_factor * energy_deficit) - (factor_deficit * (22 - previous_bmi))
        # Cap the BMI increase at 1 unit
        #delta_bmi = min(delta_bmi, 1.0)
    else:
        delta_bmi = (factor_adj * energy_deficit) - (factor_deficit * (22 - previous_bmi))

    new_bmi = previous_bmi - delta_bmi

    return new_bmi


def calculate_excess_mortality(BMIt_minus_1):
    """
    Calculate the excess mortality based on the BMI value of the previous month.
    
    Clingendael: "A formula that was empirically found to fit well with these data points was 0.00023*e((18.5‑BMIt-1)^1.36). BMIt-1 is the BMI value for the previous month. 
    This formula calculates no excess mortality at BMI values above 18.5 kg/‌ m2, 
    gradually increasing mortality levels below this, 
    and rapidly increasing mortality rates for BMI values below 13 kg/m2"

    updated formula: 0.00028*e((18.5‑BMIt-1)^1.33)

    Parameters:
    BMIt_minus_1 (float): BMI value from the previous month.

    Returns:
    float: Excess mortality value.

    NB: Conservative? Calculated based on  N Amer - "It is thus very well possible that mortality in Sudan will be higher at BMI levels between 13 and 16 kg/m2"

    """
    if BMIt_minus_1 >= 18.5:
        return 0.0
    #excess_mortality = 0.00023 * math.exp((18.5 - BMIt_minus_1) ** 1.36)
    excess_mortality = 0.00028 * math.exp((18.5 - BMIt_minus_1) ** 1.33)
    # mortality = percent, so don't return higher than 1.0:
    return min(excess_mortality, 1.0)


class CalorieDistributor:
    """
    Class for distributing caloric intake across percentiles based on population data and constraints.
    """
    def __init__(self, pop_per_percentile, total_kcal_consumption, kcal_min=700, kcal_max=1540, epsilon=0.001):
        """
        Initializes the CalorieDistributor with population data and constraints.
        """
        self.pop_per_percentile = np.array(pop_per_percentile)
        self.total_kcal_consumption = total_kcal_consumption
        self.kcal_min = kcal_min
        self.kcal_max = kcal_max
        self.epsilon = epsilon
        self.percentiles = np.arange(1, 101)  # Percentile indices from 1 to 100

    def linear_distribution(self, beta1):
        """
        Distributes calories linearly across percentiles based on the given slope beta1.
        Adjusts the distribution when necessary by capping at kcal_min or kcal_max, and raises kcal_max to 1820 if excess calories remain.
        """
        def calculate_distribution(kcal_max):
            pop = self.pop_per_percentile
            x = self.percentiles  # x_i from 1 to 100

            # Calculate beta0 range based on kcal_min and kcal_max constraints
            beta0_min = self.kcal_min - beta1 * 100
            beta0_max = kcal_max - beta1 * 1

            # Function to compute total kcal given beta0
            def total_kcal(beta0):
                y = beta0 + beta1 * x
                y = np.clip(y, self.kcal_min, kcal_max)
                return np.sum(y * pop)

            # Check if total_kcal can be matched within beta0_min and beta0_max
            total_kcal_min = total_kcal(beta0_min)
            total_kcal_max = total_kcal(beta0_max)

            if total_kcal_min - self.total_kcal_consumption > self.epsilon * self.total_kcal_consumption:
                # Insufficient calories: Cap lower percentiles at kcal_min
                def total_kcal_diff_lower(c):
                    idx1 = x <= c
                    idx2 = x > c
                    S2 = np.sum(pop[idx2])
                    S2_x = np.sum(pop[idx2] * x[idx2])

                    beta0 = (self.total_kcal_consumption - self.kcal_min * np.sum(pop[idx1]) - beta1 * S2_x) / S2
                    y2 = beta0 + beta1 * x[idx2]
                    y2 = np.clip(y2, self.kcal_min, kcal_max)
                    total_kcal = self.kcal_min * np.sum(pop[idx1]) + np.sum(y2 * pop[idx2])
                    return total_kcal - self.total_kcal_consumption

                c_lower, c_upper = 1, 99
                f_lower, f_upper = total_kcal_diff_lower(c_lower), total_kcal_diff_lower(c_upper)
                if f_lower * f_upper > 0:
                    return None  # Indicates failure to find valid distribution

                c_opt = brentq(total_kcal_diff_lower, c_lower, c_upper, xtol=0.01)
                idx1 = x <= c_opt
                idx2 = x > c_opt
                S2 = np.sum(pop[idx2])
                S2_x = np.sum(pop[idx2] * x[idx2])
                beta0 = (self.total_kcal_consumption - self.kcal_min * np.sum(pop[idx1]) - beta1 * S2_x) / S2

                y = np.zeros_like(x, dtype=float)
                y[idx1] = self.kcal_min
                y[idx2] = beta0 + beta1 * x[idx2]
                y = np.clip(y, self.kcal_min, kcal_max)

            elif self.total_kcal_consumption - total_kcal_max > self.epsilon * self.total_kcal_consumption:
                # Excess calories: Cap higher percentiles at kcal_max
                def total_kcal_diff_upper(c):
                    idx1 = x <= c
                    idx2 = x > c
                    S1 = np.sum(pop[idx1])
                    S1_x = np.sum(pop[idx1] * x[idx1])

                    beta0 = (self.total_kcal_consumption - kcal_max * np.sum(pop[idx2]) - beta1 * S1_x) / S1
                    y1 = beta0 + beta1 * x[idx1]
                    y1 = np.clip(y1, self.kcal_min, kcal_max)
                    total_kcal = np.sum(y1 * pop[idx1]) + kcal_max * np.sum(pop[idx2])
                    return total_kcal - self.total_kcal_consumption

                c_lower, c_upper = 1, 99
                f_lower, f_upper = total_kcal_diff_upper(c_lower), total_kcal_diff_upper(c_upper)
                if f_lower * f_upper > 0:
                    return None  # Indicates failure to find valid distribution

                c_opt = brentq(total_kcal_diff_upper, c_lower, c_upper, xtol=0.01)
                idx1 = x <= c_opt
                idx2 = x > c_opt
                S1 = np.sum(pop[idx1])
                S1_x = np.sum(pop[idx1] * x[idx1])
                beta0 = (self.total_kcal_consumption - kcal_max * np.sum(pop[idx2]) - beta1 * S1_x) / S1

                y = np.zeros_like(x, dtype=float)
                y[idx1] = beta0 + beta1 * x[idx1]
                y[idx2] = kcal_max
                y = np.clip(y, self.kcal_min, kcal_max)

            else:
                # Beta0 can be adjusted within beta0_min and beta0_max
                def total_kcal_diff(beta0):
                    y = beta0 + beta1 * x
                    y = np.clip(y, self.kcal_min, kcal_max)
                    return np.sum(y * pop) - self.total_kcal_consumption

                f_beta0_min, f_beta0_max = total_kcal_diff(beta0_min), total_kcal_diff(beta0_max)
                if f_beta0_min * f_beta0_max > 0:
                    return None  # Indicates failure to find valid distribution

                beta0 = brentq(total_kcal_diff, beta0_min, beta0_max, xtol=self.epsilon)
                y = beta0 + beta1 * x
                y = np.clip(y, self.kcal_min, kcal_max)

            return y

        # Try initial distribution with self.kcal_max
        y = calculate_distribution(self.kcal_max)
        if y is None or np.sum(y * self.pop_per_percentile) < self.total_kcal_consumption:
            # Excess calories: Raise kcal_max to 1820 and recalculate
            warnings.warn("Raising kcal_max to 1820 due to excess calories.")
            y = calculate_distribution(1820)

        if y is None:
            #assert False, "No valid distribution found."
            # Fallback to uniform distribution if no valid distribution was found
            warnings.warn("No valid distribution found; using fallback uniform distribution.")
            y = np.full(100, self.total_kcal_consumption / np.sum(self.pop_per_percentile))

        # Final check to ensure total kcal consumption within epsilon
        total_kcal = np.sum(y * self.pop_per_percentile)
        if abs(total_kcal - self.total_kcal_consumption) > self.epsilon * self.total_kcal_consumption:
            warnings.warn("Total kcal consumption does not match within epsilon after adjustments.")

        return y


    def piecewise_linear_distribution(self, beta1, c):
        """
        Distributes calories across percentiles using a piecewise linear function.
        The function is initially increasing (with slope beta1), then flat.

        Parameters:
        - beta1: slope of the linear segment (the rate at which calories increase per percentile)
        - c: the percentile at which the distribution becomes flat

        Returns:
        - numpy array of average caloric intake per individual for each percentile group
        """
        pop = self.pop_per_percentile
        x = self.percentiles  # x_i from 1 to 100

        # Separate indices for the two segments
        idx1 = x <= c
        idx2 = x > c

        S = np.sum(pop)
        T = np.sum((c - x[idx1]) * pop[idx1])

        # Calculate y_c based on the total kcal consumption constraint
        y_c = (self.total_kcal_consumption + beta1 * T) / S

        # Ensure y_c is within kcal_min and kcal_max
        if y_c > self.kcal_max:
            y_c = self.kcal_max
            warnings.warn("y_c adjusted to kcal_max to satisfy kcal_max constraint.")
        elif y_c < self.kcal_min:
            y_c = self.kcal_min
            warnings.warn("y_c adjusted to kcal_min to satisfy kcal_min constraint.")

        # Ensure that the minimum y_i is not below kcal_min
        y_min = y_c - beta1 * (c - np.min(x[idx1]))
        if y_min < self.kcal_min:
            beta1 = (y_c - self.kcal_min) / (c - np.min(x[idx1]))
            warnings.warn("beta1 adjusted to satisfy kcal_min constraint at the lowest percentile.")

        # Recalculate T with the adjusted beta1
        T = np.sum((c - x[idx1]) * pop[idx1])
        y_c = (self.total_kcal_consumption + beta1 * T) / S

        # Calculate the average caloric intake per percentile
        y = np.zeros_like(x, dtype=float)
        y[idx1] = y_c - beta1 * (c - x[idx1])
        y[idx2] = y_c

        # Enforce kcal_min and kcal_max
        y = np.clip(y, self.kcal_min, self.kcal_max)

        # Recompute total kcal consumption with adjusted y
        total_kcal = np.sum(y * pop)

        # Check if the total kcal matches the target within the allowable epsilon
        if abs(total_kcal - self.total_kcal_consumption) > self.epsilon * self.total_kcal_consumption:
            warnings.warn("Total kcal consumption does not match within epsilon after adjustments.")

        return y




class BMIDistribution:
    """
    Class to generate the initial BMI distribution based on one of:
    - a linear distribution with a specified top BMI value
    - a logarithmic distribution to better approximate reference data
    - a reference distribution from the provided data file.
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
        return [top_bmi - ((percentile - 1) * (12 / 99)) for percentile in self.percentiles[::-1]]

    def _assign_initial_bmi_logarithmic(self, top_bmi, bottom_bmi, alpha=0.7):
        """alpha is the exponent for the logarithmic distribution.
        For a 'flatter' curve, closer to the linear, use alpha < 1."""
        delta_bmi = top_bmi - bottom_bmi
        max_ln = np.log(100)
        return [bottom_bmi + delta_bmi * (np.log(101 - percentile) / max_ln) ** alpha for percentile in self.percentiles][::-1]

    def _assign_initial_bmi_reference(self, data_file):
        dfbmi = pd.read_excel(data_file, sheet_name="energy_intake_dist")
        bmi_init = dfbmi[dfbmi.data == "bmi_init"].values
        relvalues = bmi_init[0][1:]
        return relvalues[::-1]

    def get_bmi_distribution(self):
        return self.bmi_init



def get_months_and_days(start_year, start_month, num_months):
    # returns tuple of month, with # days
    months_and_days = []
    current_date = pd.to_datetime(f'{start_year}-{start_month}-01')

    for _ in range(num_months):
        month_str = current_date# .strftime('%b-%y')
        days_in_month = current_date.days_in_month
        months_and_days.append((month_str, days_in_month))
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


