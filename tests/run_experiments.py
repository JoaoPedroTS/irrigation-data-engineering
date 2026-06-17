import sys
import os
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.models import get_models
from models.preprocessing import get_preprocessor
from models.train_evals import train_and_evaluete
from logger import get_logger

logger = get_logger()

def run_experiments(datasets:dict[str, tuple[pd.DataFrame, list]], categorical_cols: list, target_col: str) -> pd.DataFrame:
    models = get_models()
    rows = []

    for dataset_name, (df, num_cols) in datasets.items():
        missing = [c for c in [target_col] + num_cols if c not in df.columns]
        if missing:
            logger.warning("Dataset '%s' ignorado - colunas ausentes: %s", dataset_name, missing)
            continue

        cat_cols = [c for c in categorical_cols if c in df.columns]
        preprocessor = get_preprocessor(num_cols, categorical_cols)

        logger.info("=== Dataset: %s | shpe=%s ===", dataset_name, df.shape)

        for model_name, model in models.items():
            try:
                metrics = train_and_evaluete(df, model, preprocessor, target_col)
                rows.append({
                    "dataset": dataset_name,
                    "model": model_name,
                    "accuracy": round(metrics["accuracy"], 4),
                    "f1": round(metrics["f1"], 4)
                })
            except Exception as e:
                logger.error("Erro em [%s / %s]: %s", dataset_name, model_name, e)

    return pd.DataFrame(rows)