import math
import numpy as np
import pandas as pd
import warnings



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
    if not isinstance(cereal_intake, (int, float, np.int64, np.float64)) or cereal_intake <= 0:
        raise ValueError("Cereal intake must be a positive number.")

    if not (bmi <= 60):
        raise ValueError("BMI should not exceed 60.")
    #if not (10 <= bmi <= 60):
    #    raise ValueError("BMI should be in the range of 10 to 60.")

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
    deficit = min((energy_requirement - energy_intake) / energy_requirement, 1.0) # Positive for deficit, negative for surplus, max 100%

    return deficit



def update_bmi(deficit, bmi_prev, recovery=False, factor_deficit=0.091, factor_adj=3.41316, recovery_factor=3.52):
    """
    Calculate the updated BMI based on energy deficit/surplus and previous BMI.

    Args:
        deficit (float): Energy deficit (positive for deficit, negative for surplus) as a fraction of the energy requirement.
        bmi_prev (float): The BMI of the previous month.
        recovery (bool): whether in recovery
        factor_deficit (float): Coefficient for BMI adjustment during deficit (0.0914 or 0.096).
        recovery_factor (float): Coefficient for BMI adjustment during recovery

    Returns:
        float: The updated BMI for the current month.
    """
    # Input validation
    if not isinstance(deficit, (int, float)):
        raise ValueError("Deficit must be a number.")

    # if not (10 <= bmi_prev <= 60):
    #     raise ValueError("The previous BMI should be in the range of 10 to 60.")
    if not (bmi_prev <= 60):
        raise ValueError("The previous BMI should not exceed 60.")


    if factor_deficit <= 0 or recovery_factor <= 0:
        raise ValueError("Factors should be positive numbers.")

    # Adjust BMI based on deficit or surplus
    if recovery:
        # Recovery (surplus) case
        delta_bmi = (recovery_factor * deficit) - (factor_deficit * (22 - bmi_prev))
        # Cap the BMI increase at 1 unit
        #delta_bmi = min(delta_bmi, 1.0)
    else:
        delta_bmi = (factor_adj * deficit) - (factor_deficit * (22 - bmi_prev))

    bmi_new = bmi_prev - delta_bmi

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
        return 0.0
    else:
        excess_mortality = 0.00023 * math.exp((18.5 - BMIt_minus_1) ** 1.36)
        # mortality = percent, so don't return higher than 1.0:
        return min(excess_mortality, 1.0)



class CalorieDistributor:
    def __init__(self, pop_per_percentile, total_kcal_consumption, kcal_min=500, kcal_max=1540, epsilon=0.001):
        """
        Initializes the CalorieDistributor with population data and constraints.

        Parameters:
        - pop_per_percentile: list or numpy array of populations for each percentile group (length 100)
        - total_kcal_consumption: total number of calories per day to be distributed
        - kcal_min: minimum allowable caloric intake per individual
        - kcal_max: maximum allowable caloric intake per individual
        - epsilon: allowable margin of error for total caloric consumption (as a fraction, e.g., 0.001 for 0.1%)
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
        (ie. treat percentiles as the x and average caloric consumption as the y, 
        then we have $y= beta0 + beta1 * x$ and 
        $\\int_x=1^x=100 ( \beta_0 + \beta1 * x) = total_kcal_consumption $$)

        Parameters:
        - beta1: slope of the linear distribution (the rate at which calories increase per percentile)

        Returns:
        - numpy array of average caloric intake per individual for each percentile group
        """
        pop = self.pop_per_percentile
        x = self.percentiles  # x_i from 1 to 100
        S = np.sum(pop)
        S_x = np.sum(x * pop)

        # Calculate beta0 based on the total kcal consumption constraint
        beta0 = (self.total_kcal_consumption - beta1 * S_x) / S

        # Enforce constraints on beta0 to ensure y_i are within kcal_min and kcal_max

        beta0_min = self.kcal_min - beta1 * 1
        beta0_max = self.kcal_max - beta1 * 100

        # ? Consider other behavior here, e.g. assert ??
        if beta0 < beta0_min:
            beta0 = beta0_min
            warnings.warn("beta0 adjusted to beta0_min to satisfy kcal_min constraint.")
        elif beta0 > beta0_max:
            beta0 = beta0_max
            warnings.warn("beta0 adjusted to beta0_max to satisfy kcal_max constraint.")

        # Calculate the average caloric intake per percentile
        y = beta0 + beta1 * x

        # Enforce kcal_min and kcal_max
        y = np.clip(y, self.kcal_min, self.kcal_max)

        # Recompute total kcal consumption with adjusted y
        total_kcal = np.sum(y * pop)

        # Check if the total kcal matches the target within the allowable epsilon
        if abs(total_kcal - self.total_kcal_consumption) > self.epsilon * self.total_kcal_consumption:
            warnings.warn("Total kcal consumption does not match within epsilon after adjusting beta0.")

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


