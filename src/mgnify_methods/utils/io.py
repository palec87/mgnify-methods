import os
import re
from datetime import datetime
import pandas as pd
import json
from typing import Any, Dict
from jsonapi_client import Session as APISession
from .api import get_mgnify_metadata


def fetch_analysis_metadata(folder, analysisId):
    try:
        analysis_meta = pd.read_csv(f'{folder}/{analysisId}_analysis_meta.csv').reset_index(drop=True)
    except FileNotFoundError:
        print(f"Metadata file not found: Downloading...")

        with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
            analysis_meta = map(lambda r: r.json, session.iterate(f'studies/{analysisId}/analyses'))
            analysis_meta = pd.json_normalize(analysis_meta)

        analysis_meta.to_csv(f'{folder}/{analysisId}_analysis_meta.csv', index=False)
    return analysis_meta


def fetch_samples_metadata(folder, analysisId):
    try:
        samples_meta = pd.read_csv(f'{folder}/{analysisId}_samples_meta.csv').reset_index(drop=True)
    except FileNotFoundError:
        print(f"Samples metadata file not found: Downloading...")
        samples_meta = get_mgnify_metadata(analysisId)
        samples_meta.to_csv(f'{folder}/{analysisId}_samples_meta.csv', index=False)
    return samples_meta


def import_taxonomy_summary(folder, path):
    df_tax_summary = pd.read_csv(os.path.join(folder, path), sep='\t')

    df_tax_summary.rename(columns={'#SampleID': 'taxonomy'}, inplace=True)
    df_tax_summary.set_index('taxonomy', inplace=True)
    return df_tax_summary


# ---------------------------
# IO helpers
# ---------------------------
def save_config(config: Dict[str, Any], out_dir: str, filename: str = "config.json") -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return path


def extract_first_date(x):
    if pd.isna(x):
        return None
    # find the first occurrence of YYYY-MM-DD in the string
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(x))
    return match.group(0) if match else None


def process_collection_date(metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Process the 'collection_date' column in the metadata DataFrame.
    This function converts the 'collection_date' column to datetime format,
    extracts the year, month, and day, and adds them as new columns.
    It also converts the month number to the month name (abbreviated).

    Args:
        metadata (pd.DataFrame): The metadata DataFrame containing the 'collection_date' column.

    Returns:
        pd.DataFrame: The updated metadata DataFrame with new columns for year, month, and day.
    """
    new_columns = []
    # Convert the 'collection_date' column to datetime
    before = len(metadata)

    metadata["collection_date"] = (
        metadata["collection_date"]
        .apply(extract_first_date)
        .pipe(pd.to_datetime, errors="coerce")
    )
    
    invalid_count = metadata["collection_date"].isna().sum() # Count invalids (NaT)
    metadata = metadata.dropna(subset=["collection_date"]) # Drop them

    print(f"Dropped {invalid_count} rows with invalid or missing collection_date "
        f"({before - len(metadata)} actually removed).")
    # print(metadata['collection_date'].value_counts(dropna=False))
    
    # Extract the year from the 'collection_date' column
    metadata["year"] = metadata["collection_date"].apply(
        lambda x: x.year if x is not None else None
    )
    new_columns.append("year")
    # Extract the month from the 'collection_date' column
    metadata["month"] = metadata["collection_date"].apply(
        lambda x: x.month if x is not None else None
    )

    new_columns.append("month")

    # Convert month to month name
    metadata["month_name"] = metadata["month"].apply(
        lambda x: (
            datetime.strptime(str(x), "%m").strftime("%B")[:3]
            if x is not None
            else None
        )
    )
    new_columns.append("month_name")
    # Extract the day from the 'collection_date' column
    metadata["day"] = metadata["collection_date"].apply(
        lambda x: x.day if x is not None else None
    )
    new_columns.append("day")
    return metadata, new_columns