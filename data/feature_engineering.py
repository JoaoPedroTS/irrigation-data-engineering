import numpy as np
from pandas import DataFrame

def add_new_features(df: DataFrame) -> DataFrame:
    new_df = df.copy()

    soil_humidity = {
        "Black Soil": 35,
        "Alluvial Soil": 30,
        "Sandy Soil": 20,
        "Red Soil": 28,
        "Clay Soil": 45,
        "Loam": 40
    }

    new_df["soil_moisture"] = new_df["soil_type"].map(soil_humidity) + np.random.normal(0,5, len(new_df))
    new_df["solar_radiation"] = new_df["temp"] * 2 + np.random.normal(0, 2, len(new_df))
    new_df["salinity"] = new_df["soil_type"].map({"Sandy Soil": 2, "Loam Soil": 1.5, "Clay Soil": 0.8}).fillna(1.0) + np.random.normal(0, 0.2, len(new_df))

    kc_dict = {
        "Germination": 0.7,
        "Seedling Stage": 0.8,
        "Vegetative Growth / Root or tuber Development": 0.9,
        "Flowering": 1.0,
        "Pollination": 1.1,
        "Fruit/Grain/Bulb Formation": 1.0,
        "Maturation": 0.9,
        "Harvest": 0.8
    }

    new_df = new_df["Seedling Stage"].map(kc_dict) + np.random.normal(0, 0.05, len(new_df))

    new_df["soil_moisture"] = new_df["soil_moisture"].clip(0, 100)
    new_df["salinity"] = new_df["salinity"].clip(0, 5)

    return new_df