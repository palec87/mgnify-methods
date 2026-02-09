import os
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from typing import TYPE_CHECKING
from collections import defaultdict
from mgnify_methods.taxonomy import (
    aggregate_by_taxonomic_level,
    pivot_taxonomic_data,
    prevalence_cutoff_abund,
)

from momics.taxonomy import (
    rarefy_table,
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")

# if TYPE_CHECKING:
from mgnify_methods.tables import TaxonomyTable

# def calc_rarefaction_curves(abund_table, metadata, curves):
#     for sample in tqdm(abund_table.columns):
#         # _, ratio = extract_sample_stats(metadata, sample)
#         reads = np.repeat(abund_table.index, abund_table[sample].values)
#         depths, richness = rarefaction_curve(reads)

#         campaign = metadata[metadata.index==sample]['study_tag'].values[0]
#         curves[campaign].append((depths, richness))
#     return curves


def rarefaction_curve(reads, steps=20, replicates=10):
    depths = np.linspace(1, len(reads), steps, dtype=int)
    max_depth = depths[-1]
    rng = np.random.default_rng()

    richness_reps = np.zeros((replicates, len(depths)), dtype=float)
    for rep in range(replicates):
        perm = rng.permutation(reads)
        # First occurrence index of each unique taxon in the permutation.
        _, first_idx = np.unique(perm[:max_depth], return_index=True)
        first_idx_sorted = np.sort(first_idx)
        richness_reps[rep] = np.searchsorted(first_idx_sorted, depths, side="left")

    richness = richness_reps.mean(axis=0)
    return depths, richness


def calc_rarefaction_curves(table, analysis_meta, curves, feature):
    """
    Calculate rarefaction curves for all samples in the pivot table.
    Groups curves by `feature`.
    """
    for sample in tqdm(table.columns, desc="Calculating rarefaction curves"):
        # Get feature value
        feature_val = analysis_meta.loc[sample, feature]
        
        # Get reads for this sample
        reads = np.repeat(table.index, table[sample].values)
        
        # Calculate rarefaction curve
        depths, richness = rarefaction_curve(reads)
        curves[feature_val].append((depths, richness))
    
    return curves


def calculate_min_rarefaction_depth(
    tax_level: str,
    dropna: bool, 
    taxonomy_table: TaxonomyTable,
    config: dict
) -> int:
    min_depth = 1e9
    long_df_filt = aggregate_by_taxonomic_level(
        taxonomy_table.df_filt, level=tax_level, dropna=dropna
    )
    df_filt_pivot = pivot_taxonomic_data(long_df_filt)
    df_filt_pivot = prevalence_cutoff_abund(
        df_filt_pivot, 
        percent=config['filtering']['prevalence_percent'], 
        skip_columns=0
    )
    df_filt_pivot = rarefy_table(df_filt_pivot, depth=None)
    min_depth = min(min_depth, df_filt_pivot.sum().min())
    
    logger.info(f"Minimum rarefaction depth: {int(min_depth)}")
    return int(min_depth)


def rarefy_all_taxon(taxonomy_table: TaxonomyTable, analysis_meta: pd.DataFrame, config):
    logger.info("\n=== Rarefaction Curves per Taxonomic Level ===")
    tax_level = config['taxonomy']['analysis_level']
    sample_type = config['samples']['sample_type']
    dropna = config['diversity']['alpha']['dropna']

    min_depth = config['rarefaction']['depth'] or calculate_min_rarefaction_depth(
        tax_level, 
        dropna=dropna,
        taxonomy_table=taxonomy_table,
        config=config
    )
    
    pkl_path = os.path.join(config['input']['cache_dir'], f"rarefaction_curves_{tax_level}.pkl")
    rarefied_tables = {sample_type: {}}
    
    # Check if already computed
    if os.path.exists(pkl_path):
        logger.info("Loading cached rarefaction curves...")
        with open(pkl_path, 'rb') as f:
            curves_per_feature = pickle.load(f)
        calc_rarefaction = False
    else:
        logger.info("Computing rarefaction curves...")
        calc_rarefaction = True
        curves_per_feature = {}
    
    # Process each taxonomic level
    logger.info(f"  Processing {tax_level}...")
    long_df_filt = aggregate_by_taxonomic_level(
        taxonomy_table.df_filt, level=tax_level, dropna=dropna
    )
    df_filt_pivot = pivot_taxonomic_data(long_df_filt)
    df_filt_pivot = prevalence_cutoff_abund(
        df_filt_pivot, 
        percent=config['filtering']['prevalence_percent'], 
        skip_columns=0
    )
    
    if calc_rarefaction:
        curves = defaultdict(list)
        curves = calc_rarefaction_curves(df_filt_pivot, analysis_meta, curves, feature=config['feature'])
        curves_per_feature[tax_level] = curves
    
    df_filt_pivot = rarefy_table(df_filt_pivot, depth=min_depth)
    rarefied_tables[sample_type][tax_level] = df_filt_pivot
    
    if calc_rarefaction:
        with open(pkl_path, 'wb') as f:
            pickle.dump(curves_per_feature, f)
        logger.info("Rarefaction curves cached.")

    return rarefied_tables, curves_per_feature
