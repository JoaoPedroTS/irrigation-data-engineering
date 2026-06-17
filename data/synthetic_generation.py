import sys
import os
import pandas as pd
from pandas import DataFrame
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__),".."))
from logger import get_logger

logger = get_logger(__name__)

DEFAULT_CTGAN_EPOCHS = 300
DEFAULT_TVAE_EPOCHS = 300

def get_metadata(df: DataFrame, categorical_cols: list) -> Metadata:
    metadata = Metadata.detect_from_dataframe(df)
    for col in categorical_cols:
        try:
            metadata.update_columns(col, sdtype="categorical")
        except Exception:
            pass
    return metadata

def generate_with_ctgan(df: DataFrame, categorical_cols: list, num_rows: int, epochs:int=DEFAULT_CTGAN_EPOCHS, batch_size:int=500) -> DataFrame :
    logger.info("Gerando dadoscom CTGAN | epochs=%d | num_rows=%d", epochs, num_rows)
    metadata = get_metadata(df, categorical_cols)
    synthesizer = CTGANSynthesizer(metadata, epochs=epochs, batch_size=batch_size, verbose=False)
    synthesizer.fit(df)
    result = synthesizer.sample(num_rows)
    logger.info("CTGAN concluído. Shape gerado: %s", result.shape)
    return result

def generate_with_tvae(df: DataFrame, categorical_cols: list, num_rows: int, epochs:int=DEFAULT_TVAE_EPOCHS) -> DataFrame:
    logger.info("Gerando dados com TVAE | epochs=%d | num_rows=%d", epochs, num_rows)
    metadata = get_metadata(df, categorical_cols)
    synthesizer = TVAESynthesizer(metadata, epochs=epochs)
    synthesizer.fit(df)
    result = synthesizer.sample(num_rows)
    logger.info("TVAE concluído. Shape gerado: %s", result.shape)
    return result

def generate_with_copula(df: DataFrame, categorical_cols: list, num_rows: int) -> DataFrame:
    logger.info("Gerando dados com GaussianCopula | num_rows=%d", num_rows)
    metadata = get_metadata(df, categorical_cols)
    synthesizer = GaussianCopulaSynthesizer(metadata)
    synthesizer.fit(df)
    result = synthesizer.sample(num_rows)
    logger.info("GausianCopula concluído. Shape gerado: %s", result.shape)
    return result

def generate_all(df:DataFrame, categorical_cols:list, num_rows:int, ctgan_epochs:int=DEFAULT_CTGAN_EPOCHS, tvae_epochs:int=DEFAULT_TVAE_EPOCHS, target_col:str="result") -> dict[str, DataFrame]:
    logger.info("Iniciando geração com todos os métodos ...")

    results = {
        "ctgan": generate_with_ctgan(df, categorical_cols, num_rows, epochs=ctgan_epochs),
        "tvae": generate_with_tvae(df, categorical_cols, num_rows, epochs=tvae_epochs),
        "copula": generate_with_copula(df, categorical_cols, num_rows)
    }

    logger.info("Geração completa para todos os métodos")
    return results