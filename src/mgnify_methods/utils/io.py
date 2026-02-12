import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import Any, Dict
from jsonapi_client import Session as APISession
from .api import get_mgnify_metadata

from momics.metadata import (
    extract_season,
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def fetch_analysis_metadata(folder, analysisId):
    cache_path = Path(folder) / f"{analysisId}_analysis_meta.csv"
    try:
        analysis_meta = pd.read_csv(cache_path).reset_index(drop=True)
    except FileNotFoundError:
        logger.info(f"Metadata file not found: Downloading...")

        with APISession("https://www.ebi.ac.uk/metagenomics/api/v1") as session:
            analysis_meta = map(lambda r: r.json, session.iterate(f'studies/{analysisId}/analyses'))
            analysis_meta = pd.json_normalize(analysis_meta)

        analysis_meta.to_csv(cache_path, index=False)
    return analysis_meta


def fetch_samples_metadata(folder, analysisId):
    cache_path = Path(folder) / f"{analysisId}_samples_meta.csv"
    try:
        samples_meta = pd.read_csv(cache_path).reset_index(drop=True)
    except FileNotFoundError:
        logger.info(f"Samples metadata file not found: Downloading...")
        samples_meta = get_mgnify_metadata(analysisId)
        samples_meta.to_csv(cache_path, index=False)
    return samples_meta


def import_taxonomy_summary(folder, path):
    df_tax_summary = pd.read_csv(Path(folder) / path, sep='\t')

    # df_tax_summary.rename(columns={'#SampleID': 'taxonomy'}, inplace=True)
    df_tax_summary.set_index('#SampleID', inplace=True)
    return df_tax_summary


def process_analysis_metadata(cache_folder: str, ds_dict: dict) -> pd.DataFrame:
    analysis_meta_dfs = {}
    for k, values in ds_dict.items():
        analysisId = values[0]
        analysis_meta = fetch_analysis_metadata(cache_folder, analysisId)

        analysis_meta_dfs[k] = analysis_meta

    # add study tag to each analysis metadata dataframe
    for k, df in analysis_meta_dfs.items():
        logger.info(f"Analysis {k} has {df.shape[0]} samples.")
        df['study_tag'] = k

    # concatenate all metadata dataframes
    analysis_meta = pd.concat(analysis_meta_dfs.values(), ignore_index=True)
    
    return analysis_meta


def process_samples_metadata(cache_folder: str, ds_dict: dict) -> pd.DataFrame:
    samples_meta_dfs = {}
    for k, values in ds_dict.items():
        analysisId = values[0]
        samples_meta = fetch_samples_metadata(cache_folder, analysisId)
        samples_meta_dfs[k] = samples_meta

    # add study tag to each analysis metadata dataframe
    for k, df in samples_meta_dfs.items():
        logger.info(f"Analysis {k} has {df.shape[0]} samples.")
        df['study_tag'] = k

    # concatenate all metadata dataframes
    samples_meta = pd.concat(samples_meta_dfs.values(), ignore_index=True)

    samples_meta = samples_meta.drop(
        columns=[
            'geographic location (latitude)',
            'geographic location (longitude)',
            'environment-biome',
            'environment-feature',
            'environment-material',
        ],
        errors='ignore',
    )
    if 'id' in samples_meta.columns and 'sample_id' not in samples_meta.columns:
        samples_meta = samples_meta.rename(columns={'id': 'sample_id'})
    return samples_meta


def enhance_samples_metadata(samples_meta: pd.DataFrame) -> pd.DataFrame:
    # enhance metadata
    samples_meta = samples_meta.copy()
    samples_meta.rename(columns={'collection date': 'collection_date'}, inplace=True)
    samples_meta, _ = process_collection_date(samples_meta)
    samples_meta, _ = extract_season(samples_meta)

    samples_meta['sample_type'] = samples_meta['sample-name'].apply(
        lambda x: 'euk' if isinstance(x, str) and 'Euk' in x else 'prok'
    )

    bef = samples_meta.shape[0]
    samples_meta = samples_meta[samples_meta['sample_type'].isin(['prok'])].reset_index(drop=True)
    after = samples_meta.shape[0]
    logger.info(f"Filtered samples to prokaryotes only: {bef} -> {after}")
    return samples_meta


def extract_feature(factors_df, feature, samples_meta):
    for sample in factors_df.index:
        if sample not in samples_meta.index:
            factors_df.loc[sample, feature] = 'Unknown'
            continue

        sample_meta_row = samples_meta.loc[sample, :]
        if isinstance(sample_meta_row, pd.Series):
            value = sample_meta_row.get(feature, 'Unknown')
        else:
            value = sample_meta_row[feature].iloc[0] if feature in sample_meta_row.columns else 'Unknown'
        factors_df.loc[sample, feature] = value
        
    return factors_df


def load_taxonomy_summary(ds, data_folder):
    # Load taxonomy summaries
    dfs = {}
    for k, values in ds.items():
        path = values[1]
        dfs[k] = import_taxonomy_summary(data_folder, path)
        logger.info(f"Loaded {k}: {dfs[k].shape}")

    # Merge all taxonomy dataframes
    from functools import reduce
    df_tax_summary = reduce(
        lambda left, right: pd.merge(left, right, on='#SampleID', how='outer'),
        dfs.values()
    )
    df_tax_summary = df_tax_summary.astype("Int32").fillna(0)
    logger.info(f"\nMerged taxonomy table: {df_tax_summary.shape}")
    return df_tax_summary


def filter_tax_summary(df, samples_meta):
    # Filter taxonomy table to match samples_meta
    logger.info(f'\nTaxonomy samples before filtering: {df.shape[1]}')

    df = df.loc[:, df.columns.isin(samples_meta.index)]
    logger.info(f'Taxonomy samples after filtering: {df.shape[1]}')

    # Remove taxa with all zero counts
    zero_rows = df[(df == 0).all(axis=1)]
    logger.info(f"\nRemoving {zero_rows.shape[0]} taxa with all zero counts")
    df = df.loc[~(df == 0).all(axis=1)]
    return df


def assert_taxonomy_integrity(df, samples_meta):
    # Verify data integrity
    assert len(df[df.index=='sk__Archaea']) == 1, "Missing Archaea row"
    assert len(df[df.index=='sk__Eukaryota']) == 1, "Missing Eukaryota row"
    assert len(samples_meta.index) == len(df.columns), f"Sample count mismatch, "

    lst1 = sorted(samples_meta.index.tolist())
    lst2 = sorted(df.columns.tolist())
    assert lst1 == lst2, "Sample IDs don't match between metadata and taxonomy"

    logger.info("\n✓ Data tables synchronized successfully")


def filter_number_reads(sample_total_dict,cutoff):
    to_drop = []
    for _, v in sample_total_dict.items():
        for sample, (total, _) in v.items():
            if total > cutoff:
                continue
            to_drop.append(sample)
    logger.info(f"Dropping {len(to_drop)} samples with less than {cutoff} reads: {to_drop}")
    return to_drop


# ---------------------------
# IO helpers
# ---------------------------
def save_config(config: Dict[str, Any], out_dir: str, filename: str = "config.json") -> str:
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    path = out_dir_path / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return str(path)


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

    logger.info(f"Dropped {invalid_count} rows with invalid or missing collection_date "
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
    metadata["month name"] = metadata["month"].apply(
        lambda x: (
            datetime.strptime(str(x), "%m").strftime("%B")[:3]
            if x is not None
            else None
        )
    )
    new_columns.append("month name")
    # Extract the day from the 'collection_date' column
    metadata["day"] = metadata["collection_date"].apply(
        lambda x: x.day if x is not None else None
    )
    new_columns.append("day")
    return metadata, new_columns


def align_emobon_metadata(metadata):
    metadata = metadata.copy()
    metadata.rename(columns={
        'collection date': 'collection_date',
    }, inplace=True)
    return metadata


def add_meta(samples_meta, analysis_meta, cols=None):
    if cols is None:
        cols = ['attributes.instrument-platform',
                'attributes.pipeline-version',
                'attributes.analysis-summary',
                'relationships.run.data.id']
    missing = [c for c in cols if c not in analysis_meta.columns]
    if missing:
        raise KeyError(f"Missing columns in analysis_meta: {missing}")

    meta_subset = analysis_meta[cols + ['relationships.sample.data.id']].drop_duplicates()
    merged = samples_meta.merge(
        meta_subset,
        left_on='sample_id',
        right_on='relationships.sample.data.id',
        how='left',
    )
    return merged.drop(columns=['relationships.sample.data.id'])


def merge_metadata_by_index(
    left: pd.DataFrame,
    right: pd.DataFrame,
    prefer: str = "left",
    combine_duplicates: bool = True,
) -> pd.DataFrame:
    """Merge two metadata tables by index, combining columns with the same name.

    This is effectively a row-wise concatenation with shared columns merged.
    """
    if prefer not in {"left", "right"}:
        raise ValueError("prefer must be 'left' or 'right'")

    merged = pd.concat([left, right], axis=0, sort=False)

    if combine_duplicates and merged.index.has_duplicates:
        if prefer == "left":
            merged = merged.groupby(level=0, sort=False).apply(lambda g: g.ffill().bfill().iloc[0])
        else:
            merged = merged.groupby(level=0, sort=False).apply(lambda g: g.bfill().ffill().iloc[0])
        if isinstance(merged.index, pd.MultiIndex):
            merged.index = merged.index.droplevel(1)

    return merged