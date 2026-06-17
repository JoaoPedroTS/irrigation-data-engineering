import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data.load_data import load_real_data
from config import CATEGORICAL_COLS, TARGET
from models.epoch_evaluator import evaluate_epochs, best_epoch_summary
from logger import get_logger

logger = get_logger(__name__)

NUMERICAL_COLS = ["temp", "humidity"]
TARGET_COL = TARGET[0]

MAX_EPOCHS = 500
EVAL_EVERY = 50
PATIENCE = 4
MIN_DELTA = 0.002

METRICS = [
    ("composite_score", "Composite Score", "#4f8ef7"),
    ("sdv_score", "SDV Quality", "#43c59e"),
    ("ks_score", "KS Score", "#f7a84f"),
    ("tstr_accuracy", "TSTR Accuracy", "#e05c5c"),
    ("loss", "Training Loss", "#9b7fe8"),
]

def plot_model(ax_list, results_df: pd.DataFrame, model_name: str, best: dict):
    epochs = results_df["epoch"].values

    for ax, (col, label, color) in zip(ax_list, METRICS):
        vals = results_df[col].values
        ax.plot(epochs, vals, color=color, lw=2.2, marker="o", markersize=5, label=label)

        best_ep = best["epoch"]
        if best_ep in results_df["epoch"].values:
            bval = results_df.loc[results_df["epoch"] == best_ep, col].values[0]
            ax.axvline(best_ep, color="#ff4b6e", lw=1.4, ls="--", alpha=0.7)
            ax.scatter([best_ep], [bval], color="#ff4b6e", zorder=5, s=70)

        ax.set_ylabel(label, fontsize=9)
        ax.set_xlabel("Época", fontsize=9)
        ax.set_title(f"{model_name} - {label}", fontsize=10, fontweight="bold")
        ax.grid("True", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.annotate(
            f"Melhor\nép. {int(best_ep)}",
            xy = (best_ep, bval),
            xytext = (best_ep + max(epochs) * 0.05, bval),
            fontsize = 7.5,
            color = "#ff4b6e",
            arrowprops = dict(arrowstyle="->", color="#ff4b6e", lw=1)
        )

def plot_results (all_results: dict, output_path: str = "epoch_analysis.png"):
    n_models = len(all_results)
    n_metrics = len(METRICS)

    fig = plt.figure(figsize=(6 * n_metrics, 4.5 * n_models), facecolor="#0f1117")
    fig.suptitle(
        "Análise de qualidade dos dados sintéticos por Época",
        fontsize = 16, fontweight = "bold", color = "white", y = 1.01
    )

    gs = gridspec.GridSpec(n_models, n_metrics, figure=fig, hspace=0.55, wspace=0.4)

    for row_idx, (model_name, (results_df, best)) in enumerate(all_results.items()):
        axes = [fig.add_subplot(gs[row_idx, col]) for col in range(n_metrics)]

        for ax in axes:
            ax.set_facecolor("#1a1d27")
            ax.tick_params(colors="#cccccc", labelsize=8)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333344")
            ax.yaxis.label.set_color("#cccccc")
            ax.xaxis.label.set_color("#cccccc")
            ax.title.set_color("white")

        plot_model(axes, results_df, model_name.upper(), best)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    logger.info("Gráfico salvo em: %s", output_path)
    plt.close

def run(model_type: str, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    os.makedirs("results/epochs", exist_ok=True)
    results_df = evaluate_epochs(
        df = df,
        categorical_cols = CATEGORICAL_COLS,
        numerical_cols = NUMERICAL_COLS,
        target_col = TARGET_COL,
        model_type = model_type,
        max_epochs = MAX_EPOCHS,
        eval_every = EVAL_EVERY,
        patience = PATIENCE,
        min_delta = MIN_DELTA
    )
    best = best_epoch_summary(results_df)
    csv_path = f"results/epochs/epoch_results_{model_type}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Resultados salvos em: %s", csv_path)

    return results_df, best

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", choices=["ctgan", "tvae", "both"], default="both")
    args = parser.parse_args()

    logger.info("Carregano dado reais ...")
    df = load_real_data()
    logger.info("Shape dos dados: %s" ,df.shape)

    models_to_run =["ctgan", "tvae"] if args.model == "both" else [args.model]
    all_results = {}

    for m in models_to_run:
        results_df, best = run (m, df)
        all_results[m] = (results_df, best)

    plot_results(all_results, output_path="results/epochs/epoch_analysis.png")

    logger.info("=== Recomendações Finais ===")
    for model_name, (_, best) in all_results.items():
        logger.info(
            "%s -> épocas ideais: %d (composite=%.4f)",
            model_name.upper(), int(best["epoch"]), best["composite_score"]
        )

if __name__ == "__main__":
    main()