import re
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
from typing import Any, Dict, Tuple
from jsonapi_client import Session as APISession
import requests
import xml.etree.ElementTree as ET
import time
from .api import get_mgnify_metadata

from momics.metadata import (
    extract_season,
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


# ---------------------------
# IO helpers
# ---------------------------
def save_config(config: Dict[str, Any], out_dir: str, filename: str = "config.json") -> str:
    """Save configuration dictionary to a JSON file.

    Args:
        config: Dictionary containing configuration parameters.
        out_dir: Output directory path where the config file will be saved.
        filename: Name of the output file. Defaults to "config.json".

    Returns:
        String path to the saved configuration file.
    """
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    path = out_dir_path / filename
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, sort_keys=True)
    return str(path)


def save_df(df: pd.DataFrame, out_dir: Path, filename: str, feature: str, tag: str = None) -> None:
    """Save DataFrame to CSV file with formatted filename.

    Args:
        df: DataFrame to save.
        out_dir: Output directory path.
        filename: Base filename without extension.
        feature: Feature name to include in the filename.
        tag: Optional tag to append to filename. Defaults to None.
    """
    if tag:
        out_path = out_dir / f"{filename}_{feature}_{tag}.csv"
    else:
        out_path = out_dir / f"{filename}_{feature}.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {filename} to: {out_path}")


def save_plot(fig, out_dir: Path, filename: str, feature: str, tag: str = None) -> None:
    """Save plot figure to PNG file with formatted filename.

    Args:
        fig: Matplotlib figure object to save.
        out_dir: Output directory path.
        filename: Base filename without extension.
        feature: Feature name to include in the filename.
        tag: Optional tag to append to filename. Defaults to None.
    """
    if tag:
        out_path = out_dir / f"{filename}_{feature}_{tag}.png"
    else:
        out_path = out_dir / f"{filename}_{feature}.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot {filename} to: {out_path}")


def fetch_analysis_metadata(folder: str, analysisId: str) -> pd.DataFrame:
    """Fetch analysis metadata from cache or MGnify API.

    Attempts to load metadata from a cached CSV file. If not found,
    downloads the metadata from the MGnify API and caches it.

    Args:
        folder: Directory path for caching metadata files.
        analysisId: MGnify study analysis identifier.

    Returns:
        DataFrame containing analysis metadata.
    """
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


#------------------------
# # ENA metadata
#------------------------
def parse_sample_xml(xml_bytes):
    """Parse ENA sample XML data into a DataFrame.

    Extracts sample metadata, identifiers, taxonomy information, and
    sample attributes from ENA XML format.

    Args:
        xml_bytes: Raw XML bytes from ENA API response.

    Returns:
        DataFrame with parsed sample information including accession,
        taxonomy, and all sample attributes as columns.
    """
    root = ET.fromstring(xml_bytes)

    rows = []

    for sample in root.findall(".//SAMPLE"):
        row = {}

        # --- basic metadata ---
        row["accession"] = sample.attrib.get("accession")
        row["alias"] = sample.attrib.get("alias")
        row["center_name"] = sample.attrib.get("center_name")

        row["title"] = sample.findtext("TITLE")
        row["description"] = sample.findtext("DESCRIPTION")

        # --- identifiers ---
        row["primary_id"] = sample.findtext(".//PRIMARY_ID")
        row["secondary_id"] = sample.findtext(".//SECONDARY_ID")

        # --- taxonomy ---
        row["taxon_id"] = sample.findtext(".//TAXON_ID")
        row["scientific_name"] = sample.findtext(".//SCIENTIFIC_NAME")

        # --- attributes (key-value flattening) ---
        for attr in sample.findall(".//SAMPLE_ATTRIBUTE"):
            tag = attr.findtext("TAG")
            value = attr.findtext("VALUE")
            units = attr.findtext("UNITS")

            if tag:
                col = tag.strip()

                # optionally append units
                if units:
                    row[col] = value
                    row[f"{col}_units"] = units
                else:
                    row[col] = value

        rows.append(row)

    return pd.DataFrame(rows)


def retrieve_ena_metadata(samples_meta: pd.DataFrame) -> pd.DataFrame:
    """Retrieve metadata from ENA for all biosample accessions.

    Fetches XML metadata from ENA API for each unique biosample
    accession in the input metadata and parses it into a DataFrame.

    Args:
        samples_meta: DataFrame containing 'biosample' column with
            biosample accessions.

    Returns:
        DataFrame with combined metadata from all samples, or empty
        DataFrame if no metadata could be retrieved.
    """
    dfs = []
    for biosample_accession in samples_meta['biosample'].unique():
        time.sleep(0.5)  # Be polite to the ENA API
        url = f"https://www.ebi.ac.uk/ena/browser/api/xml/{biosample_accession}"
        response = requests.get(url)
    
        if response.status_code == 200:
            logger.info(f"Successfully retrieved metadata for {biosample_accession}")
            dfs.append(parse_sample_xml(response.content))
        else:
            logger.error(f"Failed to retrieve metadata for {biosample_accession} (Status code: {response.status_code})")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()
    

def cast_numeric_columns(df: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    """Convert string columns to numeric type where possible.

    Attempts to convert each column to numeric type. Columns are converted
    only if the conversion success rate meets the threshold.

    Args:
        df: Input DataFrame to process.
        threshold: Minimum success rate (0-1) required to convert a column.
            Defaults to 1.0 (100% successful conversions required).

    Returns:
        DataFrame with numeric columns converted.
    """
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        success_rate = converted.notna().mean()

        if success_rate >= threshold:
            df[col] = converted
            logger.info(f"Casted column '{col}' to numeric (success rate: {success_rate:.2%})")

    return df


#---------------------------------------
## Import and process data and metadata
#---------------------------------------
def fetch_samples_metadata(folder: str, analysisId: str) -> pd.DataFrame:
    """Fetch sample metadata from cache or MGnify API.

    Attempts to load metadata from a cached CSV file. If not found,
    downloads the metadata using get_mgnify_metadata and caches it.

    Args:
        folder: Directory path for caching metadata files.
        analysisId: MGnify study analysis identifier.

    Returns:
        DataFrame containing sample metadata.
    """

    samples_meta_mgnify = fetch_samples_metadata_mgnify(folder, analysisId)
    samples_meta_ena = fetch_ena_metadata(samples_meta_mgnify, folder, analysisId)

    if samples_meta_ena.empty:
        logger.warning("No ENA metadata retrieved. Using MGnify metadata only.")
        return samples_meta_mgnify

    logger.info(f"mgnify shape: {samples_meta_mgnify.shape}, ena shape: {samples_meta_ena.shape}")
    if samples_meta_ena.shape[0] != samples_meta_mgnify.shape[0]:
        logger.warning("ENA metadata has fewer samples than MGnify metadata. Preferring MGnify values.")
    
    samples_meta = merge_metadata_by_index(
        samples_meta_mgnify.set_index('biosample'),
        samples_meta_ena.set_index('accession'),
        prefer="left",
    ).reset_index(drop=True)

    return samples_meta


def fetch_ena_metadata(samples_meta: pd.DataFrame, folder: str, analysisId: str) -> pd.DataFrame:
    """Fetch ENA metadata for samples in the provided metadata DataFrame.

    Args:
        samples_meta: DataFrame containing sample metadata with a 'biosample' column.
        folder: Directory path for caching metadata files.
        analysisId: MGnify study analysis identifier.

    Returns:
        DataFrame with ENA metadata for the samples, or empty DataFrame if no
        metadata could be retrieved.
    """
    if 'biosample' not in samples_meta.columns:
        logger.warning("No 'biosample' column found in samples_meta. Skipping ENA metadata retrieval.")
        return pd.DataFrame()
    
    # check the cache first
    cache_path = Path(folder) / f"{analysisId}_samples_meta_ena.csv"
    if cache_path.exists():
        ena_metadata = pd.read_csv(cache_path)
    else:
        logger.info(f"=== ENA metadata file not found: Downloading... {analysisId} ===")
        ena_metadata = retrieve_ena_metadata(samples_meta)
        ena_metadata.to_csv(cache_path, index=False)
        ena_metadata = cast_numeric_columns(ena_metadata)
    return ena_metadata


def fetch_samples_metadata_mgnify(folder: str, analysisId: str) -> pd.DataFrame:
    """Fetch sample metadata from MGnify API and cache it.

    Downloads sample metadata using get_mgnify_metadata and saves it to a
    CSV file in the specified folder.

    Args:
        folder: Directory path for caching metadata files.
        analysisId: MGnify study analysis identifier.

    Returns:
        DataFrame containing sample metadata.
    """
    cache_path = Path(folder) / f"{analysisId}_samples_meta_mgnify.csv"
    try:
        samples_meta = pd.read_csv(cache_path).reset_index(drop=True)
    except FileNotFoundError:
        logger.info(f"=== Samples metadata file not found: Downloading... {analysisId} ===")
        samples_meta = get_mgnify_metadata(analysisId)
        samples_meta.to_csv(cache_path, index=False)
    return samples_meta


def import_taxonomy_summary(folder: str, path: str) -> pd.DataFrame:
    """Import taxonomy abundance summary from TSV file.

    Reads a tab-separated taxonomy file and sets the #SampleID column
    as the index.

    Args:
        folder: Base directory path.
        path: Relative path to the taxonomy TSV file.

    Returns:
        DataFrame with taxonomy abundances indexed by #SampleID.
    """
    df_tax_summary = pd.read_csv(Path(folder) / path, sep='\t')

    # df_tax_summary.rename(columns={'#SampleID': 'taxonomy'}, inplace=True)
    df_tax_summary.set_index('#SampleID', inplace=True)
    return df_tax_summary


def process_analysis_metadata(cache_folder: str, ds_dict: dict) -> pd.DataFrame:
    """Process and combine analysis metadata from multiple studies.

    Fetches analysis metadata for each study in ds_dict, adds study tags,
    and concatenates them into a single DataFrame.

    Args:
        cache_folder: Directory path for caching metadata files.
        ds_dict: Dictionary mapping study tags to [analysisId, ...] lists.

    Returns:
        Combined DataFrame with analysis metadata from all studies,
        including a 'study_tag' column.
    """
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
    """Process and combine sample metadata from multiple studies.

    Fetches sample metadata for each study in ds_dict, adds study tags,
    removes unnecessary columns, and concatenates them into a single DataFrame.

    Args:
        cache_folder: Directory path for caching metadata files.
        ds_dict: Dictionary mapping study tags to [analysisId, ...] lists.

    Returns:
        Combined DataFrame with sample metadata from all studies,
        including a 'study_tag' column.
    """
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
    """Enhance sample metadata with derived fields.

    Processes collection dates, extracts seasonal information, identifies
    sample type (prokaryote vs eukaryote), and filters to prokaryotes only.

    Args:
        samples_meta: Input DataFrame with sample metadata.

    Returns:
        Enhanced DataFrame filtered to prokaryotic samples with additional
        columns for collection_date, season, and sample_type.
    """
    # drop columns that are not needed and may cause confusion
    cols_to_drop = ['collection_date', 'collection-date']
    samples_meta = samples_meta.drop(columns=cols_to_drop, errors='ignore')

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


def load_taxonomy_summary(ds:Dict[str, list], data_folder: str) -> pd.DataFrame:
    """Load and merge taxonomy summaries from multiple studies.

    Imports taxonomy summary files for each study and performs an outer
    merge on the #SampleID, filling missing values with 0.

    Args:
        ds: Dictionary mapping study tags to [analysisId, filepath] lists.
        data_folder: Base directory path for data files.

    Returns:
        Merged DataFrame with taxonomy abundances from all studies.
    """
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


#----------------------------
# Metadata operations
#----------------------------
def extract_feature(factors_df: pd.DataFrame, feature: str, samples_meta: pd.DataFrame) -> pd.DataFrame:
    """Extract feature values from sample metadata and add to factors DataFrame.

    For each sample in factors_df, retrieves the specified feature value
    from samples_meta and adds it as a new column.

    Args:
        factors_df: DataFrame to populate with feature values.
        feature: Name of the feature column to extract.
        samples_meta: Source DataFrame containing sample metadata.

    Returns:
        Updated factors_df with the feature column populated.
    """
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


def filter_number_reads(df: pd.DataFrame, cutoff: int) -> list:
    """Identify samples with read count below the cutoff threshold.

    Args:
        df: DataFrame with a 'total_seq_submitted' column.
        cutoff: Minimum number of reads required.

    Returns:
        List of sample IDs that fall below the cutoff threshold.
    """
    to_drop = df[df['total_seq_submitted'] <= cutoff].index.tolist()
    logger.info(f"Dropping {len(to_drop)} samples with less than {cutoff} reads: {to_drop}")
    return to_drop


def extract_first_date(x: str) -> str:
    """Extract the first date in YYYY-MM-DD format from a string.

    Args:
        x: Input string potentially containing a date.

    Returns:
        First date found in YYYY-MM-DD format, or None if no date found
        or input is NaN.
    """
    if pd.isna(x):
        return None
    # find the first occurrence of YYYY-MM-DD in the string
    match = re.search(r"\d{4}-\d{2}-\d{2}", str(x))
    return match.group(0) if match else None


def split_date_range(x: str) -> Tuple[str, str]:
    """Split a date range string into start and end dates.

    Handles date ranges in format 'YYYY-MM-DD.../YYYY-MM-DD...' by extracting
    the start date (before slash) and end date (after slash). If only a single
    date is present, both start and end are set to that date.

    Args:
        x: Input string potentially containing a date or date range.

    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format, or (None, None)
        if no dates found or input is NaN.
    """
    # Handle None/NaN values
    if x is None:
        return None, None
    
    try:
        x_str = str(x).strip()
    except:
        return None, None
    
    # Check for invalid values
    if not x_str or x_str.lower() in ('nan', 'none'):
        return None, None
    
    # Check if there's a slash indicating a date range
    if '/' in x_str:
        parts = x_str.split('/')
        # Extract dates from both parts
        start_match = re.search(r"\d{4}-\d{2}-\d{2}", parts[0])
        end_match = re.search(r"\d{4}-\d{2}-\d{2}", parts[1] if len(parts) > 1 else parts[0])
        
        start_date = start_match.group(0) if start_match else None
        end_date = end_match.group(0) if end_match else None
        
        return start_date, end_date
    else:
        # Single date - use it for both start and end
        match = re.search(r"\d{4}-\d{2}-\d{2}", x_str)
        date = match.group(0) if match else None
        return date, date


def process_collection_date(metadata: pd.DataFrame) -> pd.DataFrame:
    """Process the collection_date column in the metadata DataFrame.

    Converts the collection_date column to datetime format, handling both
    single dates and date ranges (separated by slash). For date ranges,
    creates separate 'collection_date_start' and 'collection_date' (end) columns.
    Extracts year, month, day, and month name as new columns. Rows with invalid
    or missing dates are dropped.

    Args:
        metadata: The metadata DataFrame containing the 'collection_date' column.

    Returns:
        Tuple of (updated metadata DataFrame with new columns for start/end dates,
        year, month, day, and month name; list of new column names added).
    """
    new_columns = []
    # Convert the 'collection_date' column by splitting date ranges
    before = len(metadata)

    # Split date ranges into start and end dates
    date_split = metadata["collection_date"].apply(split_date_range)
    metadata["collection_date_start"] = date_split.apply(lambda x: x[0])
    metadata["collection_date"] = date_split.apply(lambda x: x[1])
    new_columns.append("collection_date_start")
    
    # Convert to datetime
    metadata["collection_date_start"] = pd.to_datetime(
        metadata["collection_date_start"], errors="coerce"
    )
    metadata["collection_date"] = pd.to_datetime(
        metadata["collection_date"], errors="coerce"
    )
    
    invalid_count = metadata["collection_date"].isna().sum()  # Count invalids (NaT)
    metadata = metadata.dropna(subset=["collection_date"])  # Drop them

    logger.info(f"Dropped {invalid_count} rows with invalid or missing collection_date "
        f"({before - len(metadata)} actually removed).")
    
    # Extract the year from the 'collection_date' column (end date)
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


def align_emobon_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    """Align EMO-BON metadata column names to standard format.

    Renames columns to match expected naming conventions.

    Args:
        metadata: Input DataFrame with EMO-BON metadata.

    Returns:
        DataFrame with standardized column names.
    """
    metadata = metadata.copy()
    metadata.rename(columns={
        'collection date': 'collection_date',
    }, inplace=True)
    return metadata


def add_meta(samples_meta: pd.DataFrame, analysis_meta: pd.DataFrame, cols: list = None) -> pd.DataFrame:
    """Add analysis metadata columns to sample metadata.

    Merges selected columns from analysis_meta into samples_meta based on
    sample ID relationships.

    Args:
        samples_meta: DataFrame with sample metadata.
        analysis_meta: DataFrame with analysis metadata.
        cols: List of column names to merge from analysis_meta. If None,
            uses default columns for instrument platform, pipeline version,
            analysis summary, and run ID.

    Returns:
        Merged DataFrame with additional analysis metadata columns.

    Raises:
        KeyError: If any specified columns are missing from analysis_meta.
    """
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

    Performs row-wise concatenation with shared columns merged. When duplicate
    indices exist, combines them by forward/backward filling based on preference.

    Args:
        left: First DataFrame to merge.
        right: Second DataFrame to merge.
        prefer: Which DataFrame's values to prefer when both have non-null values.
            Must be either "left" or "right". Defaults to "left".
        combine_duplicates: Whether to combine duplicate indices. If False,
            duplicates are kept as-is. Defaults to True.

    Returns:
        Merged DataFrame with combined metadata.

    Raises:
        ValueError: If prefer is not "left" or "right".
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


#----------------------------
# Data integrity and filtering
#----------------------------
def filter_tax_summary(df: pd.DataFrame, samples_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter taxonomy and metadata tables to match each other.

    Filters taxonomy table to only include samples present in metadata,
    filters metadata to only include samples in taxonomy, and removes
    taxa with all zero counts.

    Args:
        df: Taxonomy abundance DataFrame (taxa as rows, samples as columns).
        samples_meta: Sample metadata DataFrame.

    Returns:
        Tuple of (filtered taxonomy DataFrame, filtered metadata DataFrame).
    """
    # Filter taxonomy table to match samples_meta
    logger.info(f'\nBefore filtering: Taxonomy: {df.shape[1]}, metadata: {samples_meta.shape[0]}')

    df = df.loc[:, df.columns.isin(samples_meta.index)]
    logger.info(f'Taxonomy samples after filtering: {df.shape[1]}')

    samples_meta = samples_meta.loc[df.columns]
    logger.info(f'Metadata samples after filtering: {samples_meta.shape[0]}')

    # Remove taxa with all zero counts
    zero_rows = df[(df == 0).all(axis=1)]
    logger.info(f"\nRemoving {zero_rows.shape[0]} taxa with all zero counts")
    df = df.loc[~(df == 0).all(axis=1)]
    return df, samples_meta


def assert_taxonomy_integrity(df: pd.DataFrame, samples_meta: pd.DataFrame) -> None:
    """Verify integrity of taxonomy and metadata tables.

    Checks that required taxonomic groups exist, sample counts match,
    and sample IDs are synchronized between tables.

    Args:
        df: Taxonomy abundance DataFrame (taxa as rows, samples as columns).
        samples_meta: Sample metadata DataFrame.

    Raises:
        AssertionError: If any integrity checks fail.
    """
    # Verify data integrity
    assert len(df[df.index=='sk__Archaea']) == 1, "Missing Archaea row"
    assert len(df[df.index=='sk__Eukaryota']) == 1, "Missing Eukaryota row"
    assert len(samples_meta.index) == len(df.columns), f"Sample count mismatch, "

    lst1 = sorted(samples_meta.index.tolist())
    lst2 = sorted(df.columns.tolist())
    assert lst1 == lst2, "Sample IDs don't match between metadata and taxonomy"

    logger.info("\n✓ Data tables synchronized successfully")


# def filter_number_reads(sample_total_dict,cutoff):
#     to_drop = []
#     for _, v in sample_total_dict.items():
#         for sample, (total, _) in v.items():
#             if total > cutoff:
#                 continue
#             to_drop.append(sample)
#     logger.info(f"Dropping {len(to_drop)} samples with less than {cutoff} reads: {to_drop}")
#     return to_drop


def evaluate_completeness(df: pd.DataFrame) -> None:
    """Evaluate completeness of metadata columns.

    Calculates and logs the percentage of non-null values for each column
    in the DataFrame.

    Args:
        df: DataFrame to evaluate.
    """
    nan_counts = df.isna().sum()
    complete = nan_counts == 0
    logger.info("\nMetadata completeness evaluation:")
    logger.info(f" === {complete.sum()} complete columns")
    logger.info(f" complete columns:{complete.index[complete].tolist()}")

    # drop the complete from the completeness calculation to focus on incomplete columns
    df = df.loc[:, ~complete]

    # columns which have less than 20% missing values
    df1 = df.loc[:, df.notna().mean() >= 0.8]
    logger.info(f" === {df1.shape[1]} columns with at least 80% completeness")
    logger.info(f" columns with at least 80% completeness: {df1.columns.tolist()}")

    # drop again to focus on the most incomplete columns
    df2 = df.loc[:, df.notna().mean() < 0.5]
    logger.info(f" === {df2.shape[1]} columns with less than 50% completeness")
    logger.info(f" columns with less than 50% completeness: {df2.columns.tolist()}")