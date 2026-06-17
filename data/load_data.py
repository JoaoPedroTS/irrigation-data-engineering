import os
import pandas as pd
from pandas import DataFrame

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_real_data(path:str=None) -> DataFrame:
    if path is None:
        path = os.path.join(PROJECT_ROOT, "archive", "cropdata_updated.csv")
    df = pd.read_csv(path)
    df = df[df["result"] != 2]
    return df