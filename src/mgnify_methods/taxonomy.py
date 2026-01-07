import pandas as pd
from typing import Any, Dict, Sequence


# ---------------------------
# Taxonomy utilities
# ---------------------------
def replace_trailing_empty_with_none(df):
    empty = (df == "")
    trailing = empty.copy()
    
    for i in range(len(df.columns) - 2, -1, -1):
        trailing.iloc[:, i] = trailing.iloc[:, i] & trailing.iloc[:, i + 1]
    
    return df.mask(trailing & empty, None)


def pivot_taxonomic_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares the taxonomic data (LSU and SSU tables) for analysis. Apart from
    pivoting.

    Normalization of the pivot is optional. Methods include:

    - **None**: no normalization.
    - **tss_sqrt**: Total Sum Scaling and Square Root Transformation.
    - **rarefy**: rarefaction to a specified depth, if None, min of sample sums is used.

    TODO: refactor scaling to a new method and offer different options.

    Args:
        df (pd.DataFrame): The input DataFrame containing taxonomic information.
        normalize (str, optional): Normalization method.
            Options: None, 'tss_sqrt', 'rarefy'. Defaults to None.
        rarefy_depth (int, optional): Depth for rarefaction. If None, uses min sample sum.
            Defaults to None.

    Returns:
        pd.DataFrame: A pivot table with taxonomic data.
    """
    if isinstance(df.index, pd.MultiIndex):
        index = df.index.names[0]
        df1 = df.reset_index()
        df1.set_index(index, inplace=True)
    else:
        df1 = df.copy()

    tax_ranks = [
        'ncbi_tax_id', 'superkingdom', 'kingdom', 'phylum', 'class',
        'order', 'family', 'genus', 'species',
    ]
    prefix_map = {
        "ncbi_tax_id": "",
        "superkingdom": "sk__",
        "kingdom": "k__",
        "phylum": "p__",
        "class": "c__",
        "order": "o__",
        "family": "f__",
        "genus": "g__",
        "species": "s__",
    }
    tax_ranks_filt = [tax for tax in tax_ranks if tax in df1.columns]
    df1["taxonomic_concat"] = df1.apply(
        lambda row: ";" + ";".join(
            f"{prefix_map[tax]}{row[tax]}" if pd.notna(row[tax]) else f"{prefix_map[tax]}"
            for tax in tax_ranks_filt
        ),
        axis=1
    )
    if 'ncbi_tax_id' in tax_ranks_filt:
        pivot_table = (
            df1.pivot_table(
                index=["ncbi_tax_id", "taxonomic_concat"],
                columns=df1.index,
                values="abundance",
            )
            .fillna(0)
            .astype(int)
        )
    else:
        pivot_table = (
            df1.pivot_table(
                index=["taxonomic_concat"],
                columns=df1.index,
                values="abundance",
            )
            .fillna(0)
            .astype(int)
        )
    return pivot_table


def pivot_taxonomic_data_new(
    df: pd.DataFrame,
    abundance_col: str = "abundance",
    tax_id_col: str = "ncbi_tax_id",
    taxonomy_ranks=None,
    concat_col: str = "taxonomic_concat",
    drop_missing_tax_id: bool = False,
    fill_missing: bool = True,
    strict: bool = False,
) -> pd.DataFrame:
    """Create a pivoted abundance matrix from long-form taxonomic data.

    This is the refactored version from the notebook.
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = [
            "superkingdom",
            "kingdom",
            "phylum",
            "class",
            "order",
            "family",
            "genus",
            "species",
        ]

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    if abundance_col not in df.columns:
        raise KeyError(f"Missing required abundance column '{abundance_col}'")

    df_work = df.copy()

    if isinstance(df_work.index, pd.MultiIndex):
        first_level_name = df_work.index.names[0] or "sample"
        df_work = df_work.reset_index(level=list(range(df_work.index.nlevels)))
        df_work = df_work.set_index(first_level_name)

    has_tax_id = tax_id_col in df_work.columns
    if strict and not has_tax_id:
        raise KeyError(f"Required taxonomic id column '{tax_id_col}' not found and strict=True")

    if drop_missing_tax_id and has_tax_id:
        df_work = df_work[~df_work[tax_id_col].isna()].copy()

    def _fill(series: pd.Series) -> pd.Series:
        return series.fillna("") if fill_missing else series

    parts = []
    if has_tax_id:
        parts.append(df_work[tax_id_col].astype(str))

    for rank in taxonomy_ranks:
        if rank not in df_work.columns:
            series_part = pd.Series(["" if fill_missing else None] * len(df_work), index=df_work.index)
        else:
            series_part = _fill(df_work[rank]).astype(str)
        prefix = "sk__" if rank == "superkingdom" else f"{rank[0]}__"
        parts.append(prefix + series_part)

    df_work[concat_col] = (
        pd.concat(parts, axis=1)
        .fillna("")
        .astype(str)
        .agg(";".join, axis=1)
    )

    df_work[abundance_col] = pd.to_numeric(df_work[abundance_col], errors="coerce").fillna(0).astype(int)

    pivot_index = [concat_col] if not has_tax_id else [tax_id_col, concat_col]

    pivot = (
        df_work.pivot_table(
            index=pivot_index,
            columns=df_work.index,
            values=abundance_col,
            aggfunc="sum",
            fill_value=0,
        )
        .astype(int)
        .sort_index()
    )

    return pivot


# newer version which works on the OOP NB, but might not work in the 0X_mgnify NB for the rarefaction part
def invert_pivot_taxonomic_data(
    pivot: pd.DataFrame,
    drop_zeros: bool = True,
    target_col: list = ["taxonomic_concat"],
) -> pd.DataFrame:
    """
    Invert the pivot table produced by `pivot_taxonomic_data` back to a long-form table.

    Args:
        pivot (pd.DataFrame): Pivot table with index=['ncbi_tax_id', 'taxonomic_concat']
                             and columns = original sample ids (index values).
        drop_zeros (bool): If True, rows with abundance == 0 are removed. Default True.

    Returns:
        pd.DataFrame: Long-form DataFrame with columns:
                      [sample_col_name, 'ncbi_tax_id', 'taxonomic_concat',
                       'superkingdom','kingdom','phylum','class','order','family','genus','species',
                       'abundance']
    """
    # input checks
    if not isinstance(pivot, pd.DataFrame):
        raise TypeError("pivot must be a pandas DataFrame")


    # Reset index so we have ncbi_tax_id and taxonomic_concat as columns
    reset = pivot.reset_index()
    # reset = pivot.copy()
    # Melt wide -> long
    melted = reset.melt(
        id_vars=[c for c in target_col if c in reset.columns],
        value_vars=[c for c in reset.columns if c not in target_col],
        var_name='source material ID',
        value_name="abundance",
    )

    # Optionally drop zeros
    if drop_zeros:
        melted = melted[melted["abundance"] != 0].copy()

    # Ensure abundance integer type where possible
    try:
        melted["abundance"] = melted["abundance"].astype(int)
    except Exception:
        # fallback to numeric if some values are non-integer
        melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")

    # return melted
    # Parse taxonomic_concat into components.
    # Expected pattern (example):
    # "12345;sk__Archaea;k__...;p__Phylum;c__Class;...;g__Genus;s__Species"
    tax_columns = [
        ("superkingdom", r"sk__"),
        ("kingdom", r"k__"),
        ("phylum", r"p__"),
        ("class", r"c__"),
        ("order", r"o__"),
        ("family", r"f__"),
        ("genus", r"g__"),
        ("species", r"s__"),
    ]

    def parse_tax_concat(s: str):
        # initialize result with empty strings
        res = {col: "" for col, _ in tax_columns}
        if pd.isna(s):
            return res
        # split by ';'
        parts = [p.strip() for p in s.split(";") if p.strip() != ""]
        # first part may be ncbi_tax_id (numeric) - but we rely on explicit ncbi_tax_id column already
        for p in parts:
            for col, prefix in tax_columns:
                if p.startswith(prefix):
                    # remove prefix and use remainder; keep empty if nothing after prefix
                    value = p[len(prefix) :].strip()
                    # normalize empty strings to NaN? Keep as empty string for now
                    res[col] = value
                    break
        return res

    # Apply parser and expand into separate columns
    parsed = melted[target_col[0]].apply(parse_tax_concat)
    parsed_df = pd.DataFrame(parsed.tolist(), index=melted.index)

    # replace trailing empty strings with None
    parsed_df = replace_trailing_empty_with_none(parsed_df)

    result = pd.concat([melted.reset_index(drop=True), parsed_df.reset_index(drop=True)], axis=1)

    # Reorder columns for readability
    cols_order = [
        "source material ID",
        "ncbi_tax_id",
        "abundance",
        "superkingdom",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
    ]
    # keep only columns that exist (in case some are missing)
    cols_order = [c for c in cols_order if c in result.columns]
    result = result[cols_order]

    # Reset index to a clean integer index
    result = result.reset_index(drop=True)
    try:
        result.set_index(['source material ID', 'ncbi_tax_id'], inplace=True)
    except KeyError:
        result.set_index(['source material ID'], inplace=True)

    result['abundance'] = pd.to_numeric(result['abundance'], errors='raise', downcast='unsigned')

    return result


def fill_lower_taxa(df: pd.DataFrame, taxonomy_ranks: list) -> pd.DataFrame:
    """
    Fill lower taxonomy ranks with None if the current rank is empty and the lower rank is also empty.
    Starts with the lowest rank and moves upwards.
    
    Args:
        df (pd.DataFrame): DataFrame with taxonomic ranks as columns.
        taxonomy_ranks (list): List of taxonomy rank column names in hierarchical order.

    Returns:
        pd.DataFrame: DataFrame with lower taxonomy ranks filled with None where appropriate.
    """
    df_out = df.copy()
    df_out[taxonomy_ranks[-1]] = df_out[taxonomy_ranks[-1]].replace('', None)
    for i in range(2, len(taxonomy_ranks)):
        lower = taxonomy_ranks[-i + 1]  # lower rank column
        current = taxonomy_ranks[-i]

        df_out[current] = df_out.apply(
        lambda row: None if (row[current] == '' and pd.isna(row[lower])) else row[current],
        axis=1
    )
    return df_out


def aggregate_by_taxonomic_level(df: pd.DataFrame, level: str, dropna: bool = True) -> pd.DataFrame:
    """
    Aggregates the DataFrame by a specific taxonomic level and sums abundances across samples.

    Args:
        df (pd.DataFrame): The input DataFrame containing taxonomic information.
        level (str): The taxonomic level to aggregate by (e.g., 'phylum', 'class', etc.).
        dropna (bool): If True, rows with NaN in the specified level and all higher levels
            are dropped before aggregation. Default is True.

    Returns:
        pd.DataFrame: A DataFrame aggregated by the specified taxonomic level.
    """
    TAXONOMY_RANKS = ['superkingdom', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
    if level not in df.columns:
        raise KeyError(f"Taxonomic level '{level}' not found in DataFrame")
    
    levels = TAXONOMY_RANKS[:TAXONOMY_RANKS.index(level)+1]

    # working = df if not dropna else df.dropna(subset=[level])
    # working = df if not dropna else df.dropna(subset=levels)
    working = df.copy()

    # Group by the specified level and sum abundances across samples (columns)
    grouped = (
        working
        .groupby([working.index.name, *levels], dropna=dropna)
        .sum(numeric_only=True)
        .reset_index()
        .set_index(working.index.name)
    )
    return grouped


# ---------------------------
# Taxonomy parsing helpers
# ---------------------------

_PREFIX_MAP = {
    "superkingdom": "sk__",
    "kingdom": "k__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}


def parse_taxonomic_concat(value: str, taxonomy_ranks: Sequence[str] | None = None) -> Dict[str, Any]:
    """Parse a semicolon-separated taxonomic_concat string into rank columns.

    Example of expected pattern:
        "sk__Bacteria;k__...;p__Firmicutes;...;g__Lactobacillus;s__acidophilus"

    Returns a dict mapping rank name -> value (empty string if missing).
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = list(_PREFIX_MAP.keys())

    result: Dict[str, Any] = {r: "" for r in taxonomy_ranks}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return result

    parts = [p.strip() for p in str(value).split(";") if p.strip()]
    for part in parts:
        for rank in taxonomy_ranks:
            pref = _PREFIX_MAP.get(rank, f"{rank[0]}__")
            if part.startswith(pref):
                result[rank] = part[len(pref):].strip()
                break
    return result


def wide_to_long_with_ranks(
    df_wide: pd.DataFrame,
    taxonomy_col: str = "taxonomy",
    abundance_col: str = "abundance",
    taxonomy_ranks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert wide table (taxonomy rows x sample columns) to long with rank columns.

    Parameters
    ----------
    df_wide : DataFrame with taxonomy strings in `taxonomy_col` and samples as columns.
    taxonomy_col : Name of the column/index holding taxonomy string (e.g., 'taxonomy').
    abundance_col : Name for the abundance value column in the long output.
    taxonomy_ranks : Optional list of ranks to extract; defaults to common ranks.

    Returns
    -------
    DataFrame with columns: sample, abundance, and one column per taxonomy rank.
    """
    if taxonomy_ranks is None:
        taxonomy_ranks = list(_PREFIX_MAP.keys())

    if df_wide.index.name == taxonomy_col or taxonomy_col in getattr(df_wide, "columns", []):
        df2 = df_wide.copy()
    else:
        # Try to reset index to find taxonomy column
        df2 = df_wide.copy()
        df2.index.name = taxonomy_col

    if taxonomy_col not in df2.columns:
        df2 = df2.reset_index()

    long_df = df2.melt(
        id_vars=[taxonomy_col],
        var_name="sample",
        value_name=abundance_col,
    )

    # Drop zeros quickly to reduce parsing cost
    long_df = long_df[long_df[abundance_col] != 0].copy()

    parsed = long_df[taxonomy_col].apply(lambda v: parse_taxonomic_concat(v, taxonomy_ranks))
    parsed_df = pd.DataFrame(parsed.tolist(), index=long_df.index)
    out = pd.concat([long_df.drop(columns=[taxonomy_col]), parsed_df], axis=1)

    # Ensure abundance numeric int
    out[abundance_col] = pd.to_numeric(out[abundance_col], errors="coerce").fillna(0).astype(int)
    return out


def prevalence_cutoff_abund(
    df: pd.DataFrame, percent: float = 10, skip_columns: int = 2, verbose: bool = True
) -> pd.DataFrame:
    """
    Apply a prevalence cutoff to the DataFrame, removing features that have abundance
    lower than `percent` in the sample. This goes sample (column) by sample independently.

    Args:
        df (pd.DataFrame): The input DataFrame containing feature abundances.
        percent (float): The prevalence threshold as a percentage.
        skip_columns (int): The number of columns to skip (e.g., taxonomic information).

    Returns:
        pd.DataFrame: A filtered DataFrame with low-prevalence features removed.
    """
    out = df.copy()
    max_threshold = 0
    for col in df.iloc[:, skip_columns:]:
        threshold = (percent / 100) * df[col].sum()

        # how many are below threshold?
        max_threshold = max(max_threshold, threshold)
        
        # set to zero those below threshold
        out.loc[df[col] < threshold, col] = 0


    # remove rows that are all zeros in the abundance columns
    out = out[(out.iloc[:, skip_columns:] != 0).any(axis=1)]
    if verbose:
        print(f"Prevalence cutoff at {percent}% (max threshold {max_threshold}): reduced from {df.shape} to {out.shape}")
    return out


def remove_singletons_per_sample(
    df: pd.DataFrame, skip_columns: int = 2, verbose: bool = True
) -> pd.DataFrame:
    """
    Remove singletons (features with count = 1) independently in each sample column.

    Args:
        df (pd.DataFrame): The input abundance DataFrame.
        skip_columns (int): Number of non-abundance columns (e.g., taxonomy info) to skip.
        verbose (bool): If True, print reduction summary.

    Returns:
        pd.DataFrame: A filtered DataFrame with singletons removed.
    """
    out = df.copy()
    before_shape = df.shape

    # For each abundance column, set singletons to zero
    for col in df.columns[skip_columns:]:
        out.loc[df[col] == 1, col] = 0

    # Remove rows that are all zero across all abundance columns
    out = out[(out.iloc[:, skip_columns:] != 0).any(axis=1)]

    if verbose:
        removed = before_shape[0] - out.shape[0]
        print(f"Removed {removed} singleton features. Shape reduced from {before_shape} → {out.shape}")

    return out
