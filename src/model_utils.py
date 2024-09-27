import math
import numpy as np
import warnings


# calculate_cereal_deficit
# update_bmi
# calculate_excess_mortality

# Energy requirements based on BMI categories
energy_requirements = {
    "BMI < 15 kg/m²": 1900,
    "15 ≤ BMI < 18.5 kg/m²": 2100,
    "BMI ≥ 18.5 kg/m²": 2200
}

def calculate_energy_deficit(cereal_intake, bmi, energy_requirements, percent_grain=0.7):
    """
    Calculate the energy deficit or surplus as a fraction of the energy requirement.

    Args:
        cereal_intake (float): The cereal intake in kcal (must be positive).
        bmi (float): The current BMI of the individual.
        energy_requirements (dict): A dictionary of energy requirements based on BMI categories.
        percent_grain (float): The percentage of calories from grains (must be between 0 and 1, default is 0.7).

    Returns:
        float: The energy deficit (positive for deficit, negative for surplus) as a fraction of the energy requirement.
    """
    # Input validation
    if not isinstance(cereal_intake, (int, float)) or cereal_intake <= 0:
        raise ValueError("Cereal intake must be a positive number.")

    if not (10 <= bmi <= 60):
        raise ValueError("BMI should be in the range of 10 to 60.")

    if bmi < 15 or bmi > 30:
        warnings.warn("Warning: extreme values of BMI (<15 or >30).")

    if not isinstance(percent_grain, (int, float)) or not (0 < percent_grain <= 1):
        raise ValueError("Percent grain must be a number between 0 and 1.")

    # Validate energy requirements dictionary
    required_bmi_categories = [
        "BMI < 15 kg/m²",
        "15 ≤ BMI < 18.5 kg/m²",
        "BMI ≥ 18.5 kg/m²"
    ]
    if not all(category in energy_requirements for category in required_bmi_categories):
        raise ValueError("The energy requirements dictionary must contain all required BMI categories.")

    # Determine BMI category
    if bmi < 15:
        bmi_category = "BMI < 15 kg/m²"
    elif 15 <= bmi < 18.5:
        bmi_category = "15 ≤ BMI < 18.5 kg/m²"
    else:
        bmi_category = "BMI ≥ 18.5 kg/m²"

    # Calculate energy requirement and intake
    energy_requirement = energy_requirements[bmi_category]
    energy_intake = cereal_intake / percent_grain

    # Calculate energy deficit or surplus fraction
    deficit = (energy_requirement - energy_intake) / energy_requirement  # Positive for deficit, negative for surplus

    return deficit



def update_bmi(deficit, bmi_prev, factor_deficit=0.096, recovery_factor=3.52):
    """
    Calculate the updated BMI based on energy deficit/surplus and previous BMI.

    Args:
        deficit (float): Energy deficit (positive for deficit, negative for surplus) as a fraction of the energy requirement.
        bmi_prev (float): The BMI of the previous month.
        factor_deficit (float): Coefficient for BMI adjustment during deficit (default is 0.096).
        recovery_factor (float): Coefficient for BMI adjustment during recovery (default is 3.52).

    Returns:
        float: The updated BMI for the current month.
    """
    # Input validation
    if not isinstance(deficit, (int, float)):
        raise ValueError("Deficit must be a number.")

    if not (10 <= bmi_prev <= 60):
        raise ValueError("The previous BMI should be in the range of 10 to 60.")

    if factor_deficit <= 0 or recovery_factor <= 0:
        raise ValueError("Factors should be positive numbers.")

    # Adjust BMI based on deficit or surplus
    if deficit >= 0:
        # Deficit case
        factor = factor_deficit
        delta_bmi = - (3.0 * deficit) - (factor * (22 - bmi_prev))
    else:
        # Recovery (surplus) case
        factor = recovery_factor
        delta_bmi = - (3.0 * deficit) + (factor * (22 - bmi_prev))
        # Cap the BMI increase at 1 unit
        delta_bmi = min(delta_bmi, 1.0)

    bmi_new = bmi_prev + delta_bmi

    return bmi_new


def calculate_excess_mortality(BMIt_minus_1):
    """
    Calculate the excess mortality based on the BMI value of the previous month.
    
    Clingendael: "A formula that was empirically found to fit well with these data points was 0.00023*e((18.5‑BMIt-1)^1.36). BMIt-1 is the BMI value for the previous month. 
    This formula calculates no excess mortality at BMI values above 18.5 kg/‌ m2, 
    gradually increasing mortality levels below this, 
    and rapidly increasing mortality rates for BMI values below 13 kg/m2"

    Parameters:
    BMIt_minus_1 (float): BMI value from the previous month.

    Returns:
    float: Excess mortality value.

    NB: Conservative? Calculated based on  N Amer - "It is thus very well possible that mortality in Sudan will be higher at BMI levels between 13 and 16 kg/m2"

    """
    if BMIt_minus_1 >= 18.5:
        return 0
    else:
        excess_mortality = 0.00023 * math.exp((18.5 - BMIt_minus_1) ** 1.36)
        return excess_mortality



# Example usage
if __name__ == "__main__":
    # Example 1: Deficit scenario
    cereal_intake = 1400  # kcal
    bmi_prev = 20.0
    percent_grain = 0.7

    deficit = calculate_energy_deficit(cereal_intake, bmi_prev, energy_requirements, percent_grain)
    bmi_new = update_bmi(deficit, bmi_prev)

    print(f"Deficit: {deficit:.2f}")
    print(f"Previous BMI: {bmi_prev}")
    print(f"Updated BMI: {bmi_new:.2f}")
    mortality = calculate_excess_mortality(bmi_prev)
    print(f"Excess mortality: {mortality:.2f}")

    # Example 2: Surplus (recovery) scenario
    cereal_intake = 1800  # kcal
    bmi_prev = 18.0
    percent_grain = 0.7

    deficit = calculate_energy_deficit(cereal_intake, bmi_prev, energy_requirements, percent_grain)
    bmi_new = update_bmi(deficit, bmi_prev)

    print(f"\nDeficit: {deficit:.2f}")
    print(f"Previous BMI: {bmi_prev}")
    print(f"Updated BMI: {bmi_new:.2f}")

    # Example 3: Extreme BMI value
    cereal_intake = 1600  # kcal
    bmi_prev = 14.0  # Extreme low BMI
    percent_grain = 0.7

    deficit = calculate_energy_deficit(cereal_intake, bmi_prev, energy_requirements, percent_grain)
    bmi_new = update_bmi(deficit, bmi_prev)
    mortality = calculate_excess_mortality(bmi_prev)
    print(f"\nDeficit: {deficit:.2f}")
    print(f"Previous BMI: {bmi_prev}")
    print(f"Updated BMI: {bmi_new:.2f}")
    print(f"Excess mortality: {mortality:.2f}")