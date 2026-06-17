import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CATEGORICAL_COLS, TARGET
from data.load_data import load_real_data
from data.synthetic_generation import generate_with_ctgan, generate_with_tvae, generate_with_copula
from models.models import get_models
from models.preprocessing import get_preprocessor
from models.train_evals import train_and_evaluete
from logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "results/fractions"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERICAL_COLS = ["temp", "humidity"]
TARGET_COL = TARGET[0]

FRACTIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

CTGAN_EPOCHS = 350
TVAE_EPOCHS = 100

def generate_augmented(df_frac: pd.DataFrame, num_rows: int) -> dict[str, pd.DataFrame]:
    augmented = {}

    generators = {
        "ctgan": lambda: generate_with_ctgan(df_frac, CATEGORICAL_COLS, num_rows, epochs=CTGAN_EPOCHS),
        "tvae": lambda: generate_with_tvae(df_frac, CATEGORICAL_COLS, num_rows, epochs=TVAE_EPOCHS),
        "copula": lambda: generate_with_copula(df_frac, CATEGORICAL_COLS, num_rows)
    }

    for name, gen_fn in generators.items():
        try:
            synth = gen_fn()
            df_aug = pd.concat([df_frac, synth], ignore_index=True)
            augmented[name] = df_aug
            logger.info(" [%s] aumentando gerado | shape=%s", name.upper(), df_aug.shape)
        except Exception as e:
            logger.error(" Falha ao gerar sintético [%s]: %s", name, e)

    return augmented

def evaluate_dataset(df: pd.DataFrame, dataset_label: str) -> list[dict]:
    models = get_models()
    preprocessor = get_preprocessor(NUMERICAL_COLS, CATEGORICAL_COLS)
    rows = []

    for model_name, model in models.items():
        try:
            metrics = train_and_evaluete(df, model, preprocessor, TARGET_COL)
            rows.append({
                "dataset": dataset_label,
                "model": model_name,
                "accuracy": round(metrics["accuracy"], 4),
                "f1": round(metrics["f1"], 4)
            })
        except Exception as e:
            logger.error("Erro [%s / %s]: %s", dataset_label, model_name, e)
    
    return rows

def run_fraction_experiments() -> pd.DataFrame:
    df_full = load_real_data()
    n_full = len(df_full)
    logger.info("Dataset completo carregado | shape=%s", df_full.shape)

    all_rows = []

    for frac in FRACTIONS:
        n_frac = max(int(n_full*frac), 100)
        df_frac = df_full.sample(n=n_frac, random_state=42).reset_index(drop=True)

        logger.info("=== Fração %.0f%% | n=%d ===", frac*100, n_frac)

        logger.info("Avaliando baseline (real) ...")
        baseline_label = f"real_{int(frac*100)}pct"
        rows_base = evaluate_dataset(df_frac, baseline_label)
        for r in rows_base:
            r.update({"fractions": frac, "n_sample": n_frac, "augmentation": "none"})
        all_rows.extend(rows_base)

        logger.info("Gerando e avaliando datasets aumentados ...")
        augmented = generate_augmented(df_frac, num_rows=n_frac)

        for method, df_aug in augmented.items():
            aug_label = f"{method}_{int(frac*100)}pct"
            rows_aug = evaluate_dataset(df_aug, aug_label)
            for r in rows_aug:
                r.update({"fraction": frac, "n_sample": n_frac, "augmentation": method})
            all_rows.extend(rows_aug)

        results_df = pd.DataFrame(all_rows)
        out_csv = os.path.join(OUTPUT_DIR, "fraction_results.csv")
        results_df.to_csv(out_csv, index=False)
        logger.info("Resultados salvos em: %s", out_csv)

    return results_df

def plot_metric_by_fraction(results_df: pd.DataFrame, metric: str = "accuracy"):
    avg = (
        results_df
        .groupby(["fraction", "augmentation"])[metric]
        .mean()
        .reset_index()
    )
 
    methods     = avg["augmentation"].unique()
    colors      = {"none": "#333333", "ctgan": "#4C72B0", "tvae": "#DD8452", "copula": "#55A868"}
    linestyles  = {"none": "--",      "ctgan": "-",       "tvae": "-",       "copula": "-"}
    labels      = {"none": "Real (baseline)", "ctgan": "Real + CTGAN",
                   "tvae": "Real + TVAE",     "copula": "Real + Cópula Gaussiana"}
 
    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        subset = avg[avg["augmentation"] == method].sort_values("fraction")
        ax.plot(
            subset["fraction"] * 100,
            subset[metric],
            marker="o", linewidth=1.8,
            color=colors.get(method, "gray"),
            linestyle=linestyles.get(method, "-"),
            label=labels.get(method, method),
        )
 
    ax.set_xlabel("Fração do Dataset Real (%)", fontsize=11)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.set_title(f"{metric.capitalize()} médio por fração — baseline vs. aumentado", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xticks([int(f * 100) for f in FRACTIONS])
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
 
    out = os.path.join(OUTPUT_DIR, f"fraction_{metric}_by_method.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Gráfico %s salvo em: %s", metric, out)
 
 
def plot_gain_heatmap(results_df: pd.DataFrame, metric: str = "accuracy"):
    avg = (
        results_df
        .groupby(["fraction", "augmentation"])[metric]
        .mean()
        .unstack("augmentation")
    )
 
    methods_aug = [m for m in ["ctgan", "tvae", "copula"] if m in avg.columns]
    gain = avg[methods_aug].subtract(avg["none"], axis=0)
 
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sns.heatmap(
        gain.T,
        annot=True, fmt=".4f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax,
        xticklabels=[f"{int(f*100)}%" for f in gain.index],
        yticklabels=["CTGAN", "TVAE", "Cópula Gaussiana"],
        annot_kws={"size": 9},
    )
    ax.set_title(f"Ganho de {metric.capitalize()} vs. Baseline Real por Fração", fontsize=12, fontweight="bold")
    ax.set_xlabel("Fração do Dataset Real")
    ax.set_ylabel("Método de Aumentação")
    fig.tight_layout()
 
    out = os.path.join(OUTPUT_DIR, f"fraction_gain_heatmap_{metric}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Heatmap de ganho salvo em: %s", out)
 
def main():
    logger.info("Iniciando experimento de frações progressivas ...")
    results_df = run_fraction_experiments()
 
    logger.info("Gerando visualizações ...")
    plot_metric_by_fraction(results_df, metric="accuracy")
    plot_metric_by_fraction(results_df, metric="f1")
    plot_gain_heatmap(results_df, metric="accuracy")
    plot_gain_heatmap(results_df, metric="f1")
 
    # Resumo: melhor método por fração
    best = (
        results_df[results_df["augmentation"] != "none"]
        .groupby(["fraction", "augmentation"])["accuracy"]
        .mean()
        .reset_index()
        .sort_values(["fraction", "accuracy"], ascending=[True, False])
        .groupby("fraction")
        .first()
        .reset_index()
        .rename(columns={"augmentation": "best_method", "accuracy": "best_accuracy"})
    )
    logger.info("Melhor método por fração:\n%s", best.to_string(index=False))
 
    out_summary = os.path.join(OUTPUT_DIR, "best_method_by_fraction.csv")
    best.to_csv(out_summary, index=False)
    logger.info("Resumo salvo em: %s", out_summary)
 
 
if __name__ == "__main__":
    main()