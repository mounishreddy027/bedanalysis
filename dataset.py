import pandas as pd
import random
from datetime import datetime, timedelta

# Define the seasonal demand function for each ward
def get_demand(day_of_year, year, base_demand, covid_effect, season_effect, constant_demand=False):
    """
    Calculates demand for a ward based on:
    - Base demand: Sets the hierarchy of ward usage.
    - Seasonal effect: Adjusts demand based on Indian seasons.
    - COVID effect: Specific modification for 2020.
    - Constant demand: If True, demand fluctuates but remains within a range, no seasonality.
    """
    week = (day_of_year - 1) // 7 + 1  # Approximate week number
    
    if constant_demand:
        # Slight random variation around the base demand, no seasonal changes
        return random.randint(base_demand - 5, base_demand + 5)
    
    if year == 2020:  # Apply COVID-specific effects
        return random.randint(
            int(base_demand * covid_effect[0]), int(base_demand * covid_effect[1])
        )
    elif week <= 9 or week >= 48:  # Winter
        return random.randint(
            int(base_demand * season_effect["winter"][0]),
            int(base_demand * season_effect["winter"][1]),
        )
    elif 23 <= week <= 36:  # Monsoon
        return random.randint(
            int(base_demand * season_effect["monsoon"][0]),
            int(base_demand * season_effect["monsoon"][1]),
        )
    elif 37 <= week <= 47:  # Post Monsoon
        return random.randint(
            int(base_demand * season_effect["post_monsoon"][0]),
            int(base_demand * season_effect["post_monsoon"][1]),
        )
    else:  # Summer
        return random.randint(
            int(base_demand * season_effect["summer"][0]),
            int(base_demand * season_effect["summer"][1]),
        )

# Define wards and their demand patterns
wards = [
    {
        "name": "Accident and Emergency",
        "base_demand": 100,
        "covid_effect": (0.6, 0.8),
        "season_effect": {"winter": (1.0, 1.0), "summer": (1.0, 1.0), "monsoon": (1.0, 1.0), "post_monsoon": (1.0, 1.0)},
        "constant_demand": True
    },
    {
        "name": "ICU",
        "base_demand": 90,
        "covid_effect": (1.8, 2.0),
        "season_effect": {"winter": (1.0, 1.2), "summer": (0.8, 1.0), "monsoon": (1.0, 1.2), "post_monsoon": (0.8, 1.0)},
        "constant_demand": False
    },
    {
        "name": "Pediatric",
        "base_demand": 80,
        "covid_effect": (0.4, 0.6),
        "season_effect": {"winter": (1.0, 1.2), "summer": (0.8, 1.0), "monsoon": (1.2, 1.4), "post_monsoon": (1.0, 1.2)},
        "constant_demand": False
    },
    {
        "name": "Maternity",
        "base_demand": 70,
        "covid_effect": (0.4, 0.6),
        "season_effect": {"winter": (0.8, 1.0), "summer": (0.8, 1.0), "monsoon": (1.0, 1.2), "post_monsoon": (0.6, 0.8)},
        "constant_demand": False
    },
    {
        "name": "General",
        "base_demand": 60,
        "covid_effect": (1.1, 1.3),
        "season_effect": {"winter": (0.6, 0.8), "summer": (0.8, 1.0), "monsoon": (1.2, 1.4), "post_monsoon": (0.6, 0.8)},
        "constant_demand": False
    },
    {
        "name": "Acute Care",
        "base_demand": 50,
        "covid_effect": (1.1, 1.3),
        "season_effect": {"winter": (1.0, 1.0), "summer": (1.0, 1.0), "monsoon": (1.0, 1.0), "post_monsoon": (1.0, 1.0)},
        "constant_demand": True
    },
    {
        "name": "Orthopedic",
        "base_demand": 40,
        "covid_effect": (0.7, 0.9),
        "season_effect": {"winter": (1.0, 1.0), "summer": (1.0, 1.0), "monsoon": (1.0, 1.0), "post_monsoon": (1.0, 1.0)},
        "constant_demand": True
    },
    {
        "name": "Executive and VVIP",
        "base_demand": 30,
        "covid_effect": (1.0, 1.0),
        "season_effect": {"winter": (1.0, 1.0), "summer": (1.0, 1.0), "monsoon": (1.0, 1.0), "post_monsoon": (1.0, 1.0)},
        "constant_demand": True
    },
    {
        "name": "Admission",
        "base_demand": 20,
        "covid_effect": (1.3, 1.5),
        "season_effect": {"winter": (1.0, 1.2), "summer": (0.8, 1.0), "monsoon": (1.0, 1.2), "post_monsoon": (0.8, 1.0)},
        "constant_demand": False
    },
]

# Initialize an empty list to store the data
data = []

# Generate the dataset for all wards from 2004 to 2023
for year in range(2004, 2024):  # From 2004 to 2023
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date
    while current_date <= end_date:
        day_of_year = current_date.timetuple().tm_yday  # Day of the year (1-365/366)
        week = (day_of_year - 1) // 7 + 1  # Approximate week number
        row = {"Date": current_date.strftime("%Y-%m-%d"), "Year": year, "Week": week}
        for ward in wards:
            row[ward["name"]] = get_demand(
                day_of_year, year, ward["base_demand"], ward["covid_effect"], ward["season_effect"], ward["constant_demand"]
            )
        data.append(row)
        current_date += timedelta(days=1)

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
df.to_csv("hospital_daily.csv", index=False)

print("Daily dataset generated and saved to 'hospital_daily.csv'.")
