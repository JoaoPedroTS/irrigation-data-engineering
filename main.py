import os
import pandas as pd
from config import CATEGORICAL_COLS, TARGET
from data.load_data import load_real_data
from data.synthetic_generation import generate_with_copula, generate_with_tvae, generate_with_ctgan
from tests.run_experiments import run_experiments
from logger import get_logger

logger = get_logger(__name__)

NUMERICAL_COLS = ["temp", "humidity"]
TARGET_COL = TARGET[0]


# Ajustar Parametros conforme resultado dos testes
CTGAN_EPOCHS = 500
TVAE_EPOCHS = 300

OUTPUT_DIR = "results/final"
SYNTH_DIR = "archive/synthetic"

def save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Salvo: %s | shape=%s", path, df.shape)

def generate_and_save(real_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = len(real_df)
    generators = {
        "ctgan": lambda: generate_with_ctgan(real_df, CATEGORICAL_COLS, n, epochs=CTGAN_EPOCHS),
        "tvae": lambda: generate_with_tvae(real_df, CATEGORICAL_COLS, n, epochs=TVAE_EPOCHS),
        "copula": lambda: generate_with_copula(real_df, CATEGORICAL_COLS, n)
    }

    datasets = {"real": real_df}

    for method, gen_fn in generators.items():
        logger.info("Gerando sintético: %s", method.upper())
        synth = gen_fn()

        synth_key = f"synthetic_{method}"
        datasets[synth_key] = synth
        save_csv(synth, os.path.join(SYNTH_DIR, f"{synth_key}.csv"))

        aug_key = f"augmented_{method}"
        aug_df = pd.concat([real_df, synth], ignore_index=True)
        datasets[aug_key] = aug_df
        save_csv(synth, os.path.join(SYNTH_DIR, f"{aug_key}.csv"))

    return datasets

def to_experiment_dict(named_dfs:dict[str, pd.DataFrame]) -> dict[str, tuple]:
    return {
        name: (df, [c for c in NUMERICAL_COLS if c in df.columns])
        for name, df in named_dfs.items()
    }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    logger.info("Carregando dados reais ...")
    real_df = load_real_data()
    logger.info("Dados reais carregados | shape=%s", real_df.shape)

    datasets = generate_and_save(real_df)
    logger.info("Total de datasets para avaliação: %d", len(datasets))
    
    results_df = run_experiments(
        datasets=to_experiment_dict(datasets),
        categorical_cols=CATEGORICAL_COLS,
        target_col=TARGET_COL
    )

    if results_df.empty:
        logger.warning("Nenhum resultado gerado")
        return
    
    summary = (
        results_df
        .sort_values(["dataset", "accuracy"], ascending=[True, False])
        .reset_index(drop=True)
    )

    best = (
        results_df
        .loc[results_df.groupby("dataset")["accuracy"].idxmax()]
        .reset_index(drop=True)
    )

    logger.info("Melhores modelos por dataset:\n%s", best.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "results_final.csv")
    best_path = os.path.join(OUTPUT_DIR, "results_best.csv")
    summary.to_csv(summary_path, index=False)
    best.to_csv(best_path, index=False)
    logger.info("Resultados completos salvos em: %s", OUTPUT_DIR)

    return summary

if __name__ == "__main__":
    main()