import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_emobon_processed(df: pd.DataFrame) -> bool:
    """
    Simple detection logic for emobon processed format.

    Arguments:
        df: DataFrame to check.

    Returns:
        bool: True if the DataFrame matches the emobon processed format, False otherwise.
    """
    if not isinstance(df.index, pd.MultiIndex):
        return False

    if df.index.names != ["source material ID", "ncbi_tax_id"]:
        return False

    expected_cols = {
        'abundance', 'superkingdom', 'kingdom', 'phylum', 'class', 'order',
        'family', 'genus', 'species'
    }

    return set(df.columns) == expected_cols


def is_emobon_raw(df: pd.DataFrame) -> bool:
    """
    Placeholder: implement your real detection logic for raw format.
    """
    return False


def is_mgnify_raw(df: pd.DataFrame) -> bool:
    """
    Detect if the DataFrame is in MGNify raw format.

    Arguments:
        df: DataFrame to check.
    
    Returns:
        bool: True if the DataFrame matches the MGNify raw format, False otherwise.
    """
    try:
        required_cols = {"#SampleID"}
        missing = required_cols - set(df.columns)
        starts_with_sk = bool(df['#SampleID'].str.startswith("sk__").all())
        unique_sample_ids = df['#SampleID'].is_unique
        return len(missing) == 0 and isinstance(df.index, pd.RangeIndex) and unique_sample_ids and starts_with_sk
    except Exception as e:
        logger.error(f"Error checking MGNify raw format: {e}")
        return False
    

def is_abundance_processed(df: pd.DataFrame) -> bool:
    """
    Simple detection logic for abundance processed format.

    Arguments:
        df: DataFrame to check.

    Returns:
        bool: True if the DataFrame matches the abundance processed format, False otherwise.
    """
    logger.debug(f"Checking abundance processed format for DataFrame with columns: {df.columns} and index: {df.index}")
    if '#SampleID' not in list(df.columns):
        return False
    return True


def is_abundance_ncbi(df: pd.DataFrame) -> bool:
    try:
        if df.columns.name != "source material ID":
            return False

        if set(df.index.names) != {"taxonomic_concat", "ncbi_tax_id"}:
            return False

        if df.select_dtypes(include="number").shape[1] != df.shape[1]:
            return False

        return True
    except Exception:
        return False
    

def is_abundance_no_ncbi(df: pd.DataFrame) -> bool:
    try:
        if df.columns.name != "source material ID":
            return False

        if set(df.index.names) != {"taxonomic_concat"}:
            return False

        if df.select_dtypes(include="number").shape[1] != df.shape[1]:
            return False

        return True
    except Exception:
        return False
