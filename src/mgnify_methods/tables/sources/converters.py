import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mgnify_raw_to_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw MGNify dataframe to standardized format.
    """
    # Rename columns
    df = df.rename(columns={"#SampleID": 'taxonomic_concat'})
    df.columns.name = "source material ID"
    df.set_index('taxonomic_concat', inplace=True)
    return df


def emobon_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise emobon processed dataframe to expected format.
    """
    # convert abundance to integers if needed
    df['abundance'] = pd.to_numeric(df['abundance'], errors='raise', downcast='unsigned')
    logger.info(f"Abundance dtype after conversion: {df['abundance'].dtype}")
    return df


def abundance_standardise(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardise abundance processed dataframe to expected format.
    """
    # Ensure all values are numeric
    df = df.apply(pd.to_numeric, errors='raise', downcast='unsigned')
    return df


def emobon_raw_to_processed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Insert your real transformation from raw → processed.
    """
    raise NotImplementedError("Conversion from emobon_raw is not implemented yet.")
