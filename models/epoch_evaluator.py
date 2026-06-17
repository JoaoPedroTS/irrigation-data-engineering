import sys
import os 
import numpy as np
import pandas as pd 
import warnings

warnings.filterwarnings("ignore")

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
from sdv.evaluation.single_table import evaluate_quality
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from typing import Callable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from logger import get_logger
from data.synthetic_generation import get_metadata
from models.preprocessing import encode_df

logger = get_logger(__name__)

def sdv_quality_score(real_df: pd.DataFrame, synth_df: pd.DataFrame, metadata: Metadata) -> float:
    try:
        report = evaluate_quality(real_df, synth_df, metadata, verbose=False)
        return float(report.get_score())
    except Exception as e:
        logger.warning("Falha ao calcular SDV quality score: %s", e)
        return float("nan")
    
def ks_score(real_df: pd.DataFrame, synth_df: pd.DataFrame, numerical_cols: list) -> float:
    scores = []
    for col in numerical_cols:
        if col in real_df.columns and col in synth_df.columns:
            _, p = ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
            scores.append(p)
    return float(np.mean(scores)) if scores else float("nan")

def tstr_score(real_df: pd.DataFrame, synth_df: pd.DataFrame, target_col: str) -> float:
    try:
        if target_col not in real_df.columns or target_col not in synth_df.columns:
            logger.warning("Coluna alvo '%s' não encontrada para TSTR", target_col)
            return float("nan")
        synth_enc = encode_df(synth_df)
        real_enc = encode_df(real_df)

        X_synth = synth_enc.drop(columns=[target_col])
        y_synth = synth_enc[target_col]
        X_real = real_enc.drop(columns=[target_col])
        y_real = real_enc[target_col]

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_synth, y_synth)
        scores = cross_val_score(clf, X_real, y_real, cv=5, scoring="accuracy")
        return float(np.mean(scores))
    except Exception as e:
        logger.warning("Falha ao calcular TSTR score: %s", e)
        return float("nan")

def composite_score(sdv: float, ks: float, tstr: float, w_sdv=0.4, w_ks=0.3, w_tstr=0.3) -> float:
    vals = [(sdv, w_sdv), (ks, w_ks), (tstr, w_tstr)]
    total_w, total_s = 0, 0
    for v, w in vals:
        if not np.isnan(v):
            total_s += v * w
            total_w += w
    return float(total_s/total_w) if total_w > 0 else float("nan")


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = -np.inf
        self.best_epoch = 0
        self.counter = 0
        self.stop = False

    def __call__(self, epoch: int, score: float) -> bool:
        if np.isnan(score):
            return False
        
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
    
def evaluate_epochs(df, categorical_cols, numerical_cols, target_col, model_type="ctgan", max_epochs=500, eval_every=50, num_rows=None, patience=5, min_delta=0.001, on_epoch_eval=None):
    num_rows = num_rows or len(df)
    metadata = get_metadata(df, categorical_cols)
    es = EarlyStopping(patience=patience, min_delta=min_delta)
    results = []

    checkpoints = list(range(eval_every, max_epochs + 1, eval_every))
    if max_epochs not in checkpoints:
        checkpoints.append(max_epochs)

    logger.info(
        "Iniciando avaliação | modelo=%s | max_epochs=%d | eval_every=%d",
        model_type.upper(), max_epochs, eval_every,
    )

    for epoch in checkpoints:
        logger.info("Treinado %s até época %d ...", model_type.upper(), epoch)

        if model_type == "ctgan":
            synth = CTGANSynthesizer(metadata, epochs=epoch, verbose=False)
        elif model_type == "tvae":
            synth = TVAESynthesizer(metadata, epochs=epoch)
        else:
            raise ValueError(f"model_type inválido: {model_type!r}")
        
        synth.fit(df)

        try:
            loss_info = synth.get_loss_values()
            last_loss = float(loss_info["Loss"].iloc[-1]) if loss_info is not None else float("nan")
        except Exception as e:
            logger.warning("Não foi possível recuperar loss na época %d: %s", epoch, e)
            last_loss = float("nan")

        synth_df = synth.sample(num_rows)

        sdv = sdv_quality_score(df, synth_df, metadata)
        ks = ks_score(df, synth_df, numerical_cols)
        tstr = tstr_score(df, synth_df, target_col)
        comp = composite_score(sdv, ks, tstr)

        metrics = {
            "epoch": epoch,
            "loss": last_loss,
            "sdv_score": sdv,
            "ks_score": ks,
            "tstr_accuracy": tstr,
            "composite_score": comp
        }

        results.append(metrics)

        if on_epoch_eval:
            on_epoch_eval(epoch, metrics)

        if es(epoch, comp):
            logger.info(
                "Early stopping ativado na época %d. Melhor época: %d (score=%.4f)",
                epoch, es.best_epoch, es.best_score,
            )
            break

    return pd.DataFrame(results)

def best_epoch_summary(results_df: pd.DataFrame) -> dict:
    idx = results_df["composite_score"].idxmax()
    best = results_df.iloc[idx].to_dict()
    logger.info("Melhor época encontrada: %s", best)
    return best