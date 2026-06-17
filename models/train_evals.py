import sys
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logger import get_logger

logger = get_logger(__name__)

def train_and_evaluete(df: pd.DataFrame, model, preprocessor, target_cols: str) -> dict:
    X = df.drop(columns=[target_cols])
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    logger.info("Treinando %s ...", type(model.estimator if hasattr(model, "estimator") else model).__name__)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted"),
        "confusion_matrix": confusion_matrix
    }

    logger.info("Resultado | accuracy=%.4f | f1=%.4f", results["accuracy"], results["f1"])

    return results