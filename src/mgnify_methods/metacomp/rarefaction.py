from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
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


def calculate_min_rarefaction_depth(abundance_df: pd.DataFrame) -> int:
    min_depth = 1e9

    abundance_df = rarefy_table(abundance_df, depth=None)
    min_depth = int(min(min_depth, abundance_df.sum().min()))
    
    logger.info(f"Minimum rarefaction depth: {min_depth}")
    return min_depth


def rarefy_taxon(abundance_df: pd.DataFrame, config):
    logger.info("\n=== Rarefaction Curves per Taxonomic Level ===")
    tax_level = config['taxonomy']['analysis_level']

    min_depth = (
        config.get('preprocess', {})
        .get('method_params', {})
        .get('rarefaction', {})
        .get('depth')
    )

    # pkl_path = Path(config['input']['cache_dir']) / f"rarefaction_curves_{tax_level}.pkl"
    # # Check if already computed
    # if pkl_path.exists():
    #     logger.info("Loading cached rarefaction curves...")
    #     with open(pkl_path, 'rb') as f:
    #         curves_per_feature = pickle.load(f)
    #     calc_rarefaction = False
    # else:
    #     logger.info("Computing rarefaction curves...")
    #     calc_rarefaction = True
    #     curves_per_feature = {}
    
    # if calc_rarefaction:
    #     curves = defaultdict(list)
    #     curves = calc_rarefaction_curves(df_filt_pivot, analysis_meta, curves, feature=config['feature'])
    #     curves_per_feature[tax_level] = curves
    
    # df_filt_pivot = rarefy_table(df_filt_pivot, depth=min_depth)
    rarefied_tables = {tax_level: rarefy_table(abundance_df, depth=min_depth)}
    
    # if calc_rarefaction:
    #     with open(pkl_path, 'wb') as f:
    #         pickle.dump(curves_per_feature, f)
    #     logger.info("Rarefaction curves cached.")

    return rarefied_tables
