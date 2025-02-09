""" Currently includes the (modified) Clingendael-style resource scarcity model."""

import warnings
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd

from src.model_utils import (
    energy_requirements,
    BMIDistribution,
    CalorieDistributor,
    calculate_energy_deficit,
    update_bmi,
    calculate_excess_mortality,
    BMIFactors
)


class ResourceScarcityModel:
    """A model to simulate resource scarcity effects on a population."""

    def __init__(self, config):
        """Initialize the model with a configuration dictionary."""
        self.config = config

        # Create one BMIFactors object (from config file else use default)
        self.bmi_factors = BMIFactors(
            factor_deficit=self.config.get('factor_deficit', 0.091),
            factor_adj=self.config.get('factor_adj', 3.41316),
            recovery_factor=self.config.get('recovery_factor', 3.52)
        )

        # Population and grain parameters
        self.total_pop = self.config['total_pop']
        self.grain_percentage = self.config['grain_percentage']
        self.grain_stock = [
            x * self.config['grain_multiplier'] for x in self.config['grain_stock']
        ]

        # Initialize simulation data
        self.current_stock = 0
        self.monthly_values = []
        self.percentile_values = []
        self.months_and_days = []
        self.months = []
        self.days_p_month = []

        self.initialize_simulation()
        self.initialize_bmi_distribution()
        self.initialize_percentile_groups()

        # Predefine attributes that are later set in various methods.
        self.monthly_input = 0
        self.total_available = 0
        self.total_calories_this_month = 0
        self.total_cons_kcal_per_day = 0
        self.consumption = 0
        self.closing_stock = 0
        self.total_deaths_excess_mortality = 0
        self.total_deaths_due_to_bmi = 0
        self.total_deaths = 0
        self.total_natural_deaths = 0

    def initialize_simulation(self):
        """Load simulation data and parameters from files."""
        self.base_path = Path.cwd()
        self.data_file = self.base_path / 'data' / 'processed' / 'Combined_2024-10-30.xlsx'

        # Load ref data
        self.ref_data = pd.read_excel(self.data_file, sheet_name='Sc4')
        self.ref_data['month'] = pd.to_datetime(self.ref_data['month-year'], format='%m/%y')

        # Time config
        self.months_and_days = self.get_months_and_days(
            self.config['start_year'],
            self.config['start_month'],
            self.config['num_months']
        )
        self.months = [m for m, _ in self.months_and_days]
        self.days_p_month = [d for _, d in self.months_and_days]

        # Monthly total consumption
        self.monthly_total_consumption = pd.DataFrame({
            'month': self.months,
            'total_demand': [
                demand * self.config['grain_multiplier']
                for demand in self.config['monthly_total_demand']
            ]
        })

        # Read population changes data
        self.population_changes = pd.read_excel(
            self.data_file, sheet_name='population'
            )
        self.population_changes['month'] = pd.to_datetime(
            self.population_changes['month'], format='%m/%y'
            )
        self.monthly_total_consumption['month'] = pd.to_datetime(
            self.monthly_total_consumption['month']
            )

    def initialize_percentile_groups(self):
        """Initialize percentile-based population distribution."""
        percentiles = np.arange(1, 101)
        pop_per_group = self.total_pop / 100  # 1% per group

        self.percentile_groups = pd.DataFrame({
            'percentile': percentiles,
            'bmi': self.bmi_init,
            'pop': pop_per_group,
            'alive': True
        })

    def get_months_and_days(self, start_year, start_month, num_months):
        """Get the months and days for the simulation."""
        months_and_days = []
        start_date = datetime(start_year, start_month, 1)
        for i in range(num_months):
            month_date = start_date + relativedelta(months=i)
            days_in_month = (month_date + relativedelta(months=1) - month_date).days
            months_and_days.append((month_date, days_in_month))
        return months_and_days

    def initialize_bmi_distribution(self):
        """Initialize the BMI distribution for the population."""
        bmi_init_method = self.config['bmi_init_method']
        if bmi_init_method == 'linear':
            bmi_linear = BMIDistribution(
                method='linear', top_bmi=self.config['top_bmi']
                )
            self.bmi_init = bmi_linear.get_bmi_distribution()
        elif bmi_init_method == 'logarithmic':
            bmi_logarithmic = BMIDistribution(
                method='logarithmic',
                top_bmi=self.config['top_bmi']
                )
            self.bmi_init = bmi_logarithmic.get_bmi_distribution()
        elif bmi_init_method == 'reference':
            bmi_reference = BMIDistribution(
                method='reference',
                data_file=self.data_file
                )
            self.bmi_init = bmi_reference.get_bmi_distribution()
        else:
            raise ValueError(
                "Invalid BMI method/not implemented. Choose 'linear' or 'reference'."
                )

    def run_simulation(self):
        """Run the simulation for the specified number of months."""
        for i, month in enumerate(self.months):
            print(f"Processing month: {month.strftime('%Y-%m')}")
            self.process_month(i, month)

    def process_month(self, i, month):
        """Process the simulation for a single month."""
        self.update_grain_stock(i)
        self.calculate_total_calories(i, month)
        self.distribute_calories(i)
        self.update_bmi_values()
        self.calculate_mortality()
        self.update_population(month)
        self.record_monthly_values(month)
        self.record_percentile_values(month)

    def update_grain_stock(self, i):
        """Update the grain stock for the current month."""
        self.monthly_input = self.grain_stock[i]
        self.total_available = self.current_stock + self.monthly_input

    def calculate_total_calories(self, i, month):
        """Calculate the total calories available and consumed per day."""
        total_supply_consume_this_month = self.monthly_total_consumption.loc[
            self.monthly_total_consumption['month'] == month, 'total_demand'].values[0]
        self.total_calories_this_month = total_supply_consume_this_month * self.config['calories_per_metric_ton']
        self.total_cons_kcal_per_day = self.total_calories_this_month / self.days_p_month[i]

    def distribute_calories(self, i):
        """Distribute calories across the population."""
        pop_array = self.percentile_groups['pop'].values
        curr_cal_config = {
            'pop_per_percentile': pop_array,
            'total_kcal_consumption': self.total_cons_kcal_per_day,
            'kcal_min': self.config['distrib_min_kcal'],
        }
        distributor = CalorieDistributor(curr_cal_config)
        distrib_method = self.config['distrib_method']
        if distrib_method == 'linear':
            kcal_distrib = distributor.linear_distribution(beta1=self.config['distrib_beta1'])
        elif distrib_method == 'piecewise_linear':
            kcal_distrib = distributor.piecewise_linear_distribution(
                beta1=self.config['distrib_beta1'], c=self.config['distrib_c']
                )
        else:
            raise ValueError(
                "Distribution method not implemented. Choose 'linear' or 'piecewise_linear'."
                )

        self.percentile_groups['kcal_distrib'] = kcal_distrib

        kcal_consumption = (kcal_distrib * pop_array * self.days_p_month[i]).sum()
        self.consumption = kcal_consumption / self.config['calories_per_metric_ton']
        self.closing_stock = self.total_available - self.consumption
        self.current_stock = self.closing_stock

    def update_bmi_values(self):
        """Update the BMI values for the population."""
        alive_mask = self.percentile_groups['alive']

        self.percentile_groups.loc[alive_mask, 'deficit'] = self.percentile_groups.loc[alive_mask].apply(
            lambda row: calculate_energy_deficit(
                intake_kcal=row['kcal_distrib'],
                bmi=row['bmi'],
                energy_req_dict=energy_requirements,
                grain_fraction=self.grain_percentage
            ),
            axis=1
        )

        self.percentile_groups.loc[alive_mask, 'bmi'] = self.percentile_groups.loc[alive_mask].apply(
            lambda row: update_bmi(
                energy_deficit=row['deficit'],
                previous_bmi=row['bmi'],
                recovery=False,
                factors=self.bmi_factors  # single object now
            ),
            axis=1
        )


    def calculate_mortality(self):
        """Calculate the mortality due to BMI and excess mortality."""
        critical_bmi = self.config['critical_bmi']
        dead_mask = self.percentile_groups['bmi'] <= critical_bmi
        deaths_due_to_bmi = self.percentile_groups.loc[dead_mask, 'pop'].sum()
        self.percentile_groups.loc[dead_mask, 'alive'] = False
        self.percentile_groups.loc[dead_mask, 'pop'] = 0

        alive_mask = self.percentile_groups['alive']

        self.percentile_groups.loc[alive_mask, 'excess_mortality_rate'] = self.percentile_groups.loc[alive_mask]['bmi'].apply(
            calculate_excess_mortality
        )
        self.percentile_groups.loc[alive_mask, 'deaths_excess_mortality'] = np.floor(
            self.percentile_groups.loc[alive_mask, 'excess_mortality_rate'] * self.percentile_groups.loc[alive_mask, 'pop']
        )

        self.percentile_groups.loc[alive_mask, 'pop'] -= self.percentile_groups.loc[alive_mask, 'deaths_excess_mortality']
        self.percentile_groups.loc[self.percentile_groups['pop'] < 0, 'pop'] = 0

        self.total_deaths_excess_mortality = self.percentile_groups['deaths_excess_mortality'].sum()
        self.total_deaths_due_to_bmi = deaths_due_to_bmi
        self.total_deaths = self.total_deaths_due_to_bmi + self.total_deaths_excess_mortality

    def update_population(self, month):
        """ Get the population changes for the current month. """
        month_population_changes = self.population_changes[self.population_changes['month'] == month]

        if not month_population_changes.empty:
            # Get the values for births, natural_deaths, migration
            births = month_population_changes['births'].values[0]
            natural_deaths = month_population_changes['deaths'].values[0]
            migration = month_population_changes['migration'].values[0]
        else:
            warnings.warn(f"No population data found for month: {month.strftime('%Y-%m')}. Assuming no population changes.")
            births = 0
            natural_deaths = 0
            migration = 0

        # Distribute births, natural deaths, migration evenly among alive percentiles
        alive_mask = self.percentile_groups['alive']
        num_alive_percentiles = alive_mask.sum()

        if num_alive_percentiles > 0:
            # Distribute the population change evenly among alive percentiles
            pop_increase_per_percentile =  births / num_alive_percentiles
            pop_decrease_per_percentile = (natural_deaths + migration) / num_alive_percentiles

            # Update population
            self.percentile_groups.loc[alive_mask, 'pop'] += pop_increase_per_percentile
            self.percentile_groups.loc[alive_mask, 'pop'] -= pop_decrease_per_percentile

            # Ensure population does not go negative
            self.percentile_groups.loc[self.percentile_groups['pop'] < 0, 'pop'] = 0
        else:
            pass

        # Record natural deaths
        self.total_natural_deaths = natural_deaths

    def record_monthly_values(self, month):
        """Record the monthly values for the simulation."""
        monthly_record = {
            'month': month,
            'opening_stock': self.total_available,
            'monthly_input': self.monthly_input,
            'total_available': self.total_available,
            'consumption': self.consumption,
            'closing_stock': self.closing_stock,
            'total_deaths_due_to_bmi': self.total_deaths_due_to_bmi,
            'total_deaths_excess_mortality': self.total_deaths_excess_mortality,
            'natural_deaths': self.total_natural_deaths,
            'total_deaths': self.total_deaths_due_to_bmi + 
                self.total_deaths_excess_mortality + self.total_natural_deaths,
            'population': self.percentile_groups['pop'].sum()
        }
        self.monthly_values.append(monthly_record)

    def get_results(self):
        """Return the simulation results as a DataFrame."""
        return pd.DataFrame(self.monthly_values)

    def record_percentile_values(self, month):
        """Record the percentile values for the simulation."""
        percentile_data = self.percentile_groups.copy()
        percentile_data['month'] = month
        self.percentile_values.append(percentile_data)

    def get_percentile_data(self):
        """Return the percentile data as a DataFrame."""
        return pd.concat(self.percentile_values, ignore_index=True)

