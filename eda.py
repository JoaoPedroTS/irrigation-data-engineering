import os 
import sys 
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import CATEGORICAL_COLS, TARGET
from data.load_data import load_real_data
from logger import get_logger

logger = get_logger(__name__)

OUTPUT_DIR = "results/eda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUMERICAL_COLS = ["temp", "humidity"]
TARGET_COL = TARGET[0]

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = df[NUMERICAL_COLS].agg(["mean", "std", "min", "max", "median"]).T
    stats["skewness"] = df[NUMERICAL_COLS].apply(skew)
    stats["kurtosis"] = df[NUMERICAL_COLS].apply(kurtosis)
    stats["missing"] = df[NUMERICAL_COLS].isnull().sum()
    stats["missing_%"] = (stats["missing"] / len(df) * 100).round(2)
    stats = stats.round(4)
    out = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    stats.to_csv(out)
    logger.info(f"Sumário estatístico salvo em: {out}")
    return stats

def plot_class_balance(df: pd.DataFrame):
    counts = df[TARGET_COL].value_counts().sort_index()
    pct = (counts / len(df) * 100).round(2)

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index.astype(str), counts.values, color=["#4C72B0", "#DD8452"], edgecolor="black", linewidth=0.6)
    for bar, p in zip(bars, pct.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{p:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribuição da Variável Alvo (result)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Classe")
    ax.set_ylabel("Contagem")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["0 - Não irrigar", "1 - Irrigar"])
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "class_balance.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Gráfico de balanceamento salvo em: {out}")
    logger.info(f"Distribuição da variável alvo: {counts.to_string()}")

    return counts

def plot_numerical_distributions(df: pd.DataFrame):
    n = len(NUMERICAL_COLS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
 
    for i, col in enumerate(NUMERICAL_COLS):
        ax = axes[i]
        for cls, color in zip([0, 1], ["#4C72B0", "#DD8452"]):
            subset = df[df[TARGET_COL] == cls][col].dropna()
            ax.hist(subset, bins=40, alpha=0.6, label=f"Classe {cls}", color=color, edgecolor="none")
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xlabel("Valor")
        ax.set_ylabel("Frequência")
        ax.legend(fontsize=8)
 
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    fig.suptitle("Distribuição das Features Numéricas por Classe", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "numerical_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribuições numéricas salvas em: %s", out)

def plot_boxplots(df: pd.DataFrame):
    n = len(NUMERICAL_COLS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
 
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()
 
    for i, col in enumerate(NUMERICAL_COLS):
        ax = axes[i]
        data = [df[df[TARGET_COL] == cls][col].dropna().values for cls in [0, 1]]
        bp = ax.boxplot(data, patch_artist=True, widths=0.5,
                        medianprops=dict(color="black", linewidth=2))
        colors = ["#4C72B0", "#DD8452"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_xticklabels(["Classe 0", "Classe 1"])
        ax.set_ylabel("Valor")
 
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
 
    fig.suptitle("Boxplots das Features por Classe", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "boxplots_by_class.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Boxplots salvos em: {out}")

def plot_correlation_matrix(df: pd.DataFrame):
    corr = df[NUMERICAL_COLS + [TARGET_COL]].corr(method="pearson")
 
    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 9}
    )
    ax.set_title("Matriz de Correlação de Pearson", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "correlation_matrix.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Matriz de correlação salva em: %s", out)
    return corr

def plot_categorical_distributions(df: pd.DataFrame):
    valid_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
    if not valid_cats:
        logger.warning("Nenhuma coluna categórica encontrada no dataframe.")
        return
 
    n = len(valid_cats)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
 
    for ax, col in zip(axes, valid_cats):
        order = df[col].value_counts().index
        counts = df[col].value_counts()[order]
        ax.bar(range(len(order)), counts.values, color="#4C72B0", edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=9)
        ax.set_title(col, fontsize=11, fontweight="bold")
        ax.set_ylabel("Contagem")
 
    fig.suptitle("Distribuição das Features Categóricas", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "categorical_distributions.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Distribuições categóricas salvas em: %s", out)

def outlier_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in NUMERICAL_COLS:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        rows.append({
            "feature": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "lower_fence": round(lower, 4),
            "upper_fence": round(upper, 4),
            "n_outliers": n_out,
            "outlier_%": round(n_out / len(df) * 100, 2)
        })
    report = pd.DataFrame(rows).set_index("feature")
    out = os.path.join(OUTPUT_DIR, "outlier_report.csv")
    report.to_csv(out)
    logger.info("Relatório de outliers salvo em: %s", out)
    logger.info("Outliers por feature:\n%s", report[["n_outliers", "outlier_%"]].to_string())
    return report

def run_eda():
    logger.info("Iniciando EDA ...")
    df = load_real_data()
    logger.info(f"Dataset carregado | shape = {df.shape}")

    logger.info("-- [1/6] Sumário estatístico")
    stats = summary_stats(df)
    print("\nSumário Estatístico:")
    print(stats.to_string())

    logger.info("-- [2/6] Balanceamento da variável alvo")
    plot_class_balance(df)

    logger.info("-- [3/6] Distribuições numéricas")
    plot_numerical_distributions(df)

    logger.info("-- [4/6] Boxplots por classe")
    plot_boxplots(df)

    logger.info("-- [5/6] Matrix de correlação")
    corr = plot_correlation_matrix(df)

    logger.info("-- [6/6] Distribuições categóricas + outliers")
    plot_categorical_distributions(df)
    outlier_report(df)

    logger.info(f"EDA concluida. Resultado em: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_eda()