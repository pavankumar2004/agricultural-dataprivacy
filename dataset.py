"""
Dataset Generator
"""

import numpy as np
import pandas as pd

SEED = 42

NUM_FARMS = 500
REGIONS = ["North", "South", "East", "West", "Central"]
CROP_TYPES = ["Wheat", "Rice", "Corn", "Soybean"]

CROP_PARAMS = {
    "Wheat":   (3.5, 250),
    "Rice":    (4.2, 300),
    "Corn":    (5.0, 200),
    "Soybean": (2.8, 350),
}

REGION_CLIMATE = {
    "North":   (600,  18, 0.95),
    "South":   (900,  28, 1.10),
    "East":    (750,  22, 1.00),
    "West":    (400,  25, 0.85),
    "Central": (700,  20, 1.05),
}


def generate_dataset(num_farms: int = NUM_FARMS, seed: int = SEED) -> pd.DataFrame:
    """Generate dataset."""
    rng = np.random.default_rng(seed)

    records = []
    for i in range(num_farms):
        region = rng.choice(REGIONS)
        crop = rng.choice(CROP_TYPES)

        base_yield, rev_per_ton = CROP_PARAMS[crop]
        avg_rain, avg_temp, yield_mult = REGION_CLIMATE[region]

        area = round(rng.uniform(5, 200), 1)                       
        soil_moisture = round(rng.uniform(15, 45), 1)               
        rainfall = round(rng.normal(avg_rain, avg_rain * 0.15), 1)  
        temperature = round(rng.normal(avg_temp, 3), 1)             
        fertilizer = round(rng.uniform(50, 300), 1)                 

        noise = rng.normal(1.0, 0.12)
        moisture_factor = 0.8 + 0.4 * (soil_moisture - 15) / 30  
        yield_per_ha = base_yield * yield_mult * moisture_factor * noise
        total_yield = round(max(area * yield_per_ha, 0), 2)

        price_noise = rng.normal(1.0, 0.05)
        revenue = round(total_yield * rev_per_ton * price_noise, 2)

        records.append({
            "farm_id":        f"FARM-{i+1:04d}",
            "region":         region,
            "crop_type":      crop,
            "area_hectares":  area,
            "yield_tons":     total_yield,
            "soil_moisture":  soil_moisture,
            "rainfall_mm":    max(rainfall, 0),
            "temperature_c":  temperature,
            "fertilizer_kg":  fertilizer,
            "revenue_usd":    max(revenue, 0),
        })

    return pd.DataFrame(records)


def get_region_data(df: pd.DataFrame) -> dict:
    """Split dataset by region."""
    return {region: group.reset_index(drop=True)
            for region, group in df.groupby("region")}


if __name__ == "__main__":
    df = generate_dataset()
    print(f"Generated {len(df)} farm records across {df['region'].nunique()} regions")
    print(f"Crop types: {sorted(df['crop_type'].unique())}\n")
    print(df.describe().round(2))
    print(f"\nSample rows:\n{df.head(10).to_string(index=False)}")
