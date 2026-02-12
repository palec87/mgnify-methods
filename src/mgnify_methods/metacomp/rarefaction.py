from pathlib import Path
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

from momics.taxonomy import (
    rarefy_table,
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def rarefaction_curve(reads, steps=20, replicates=10):
    if len(reads) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)

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


def calc_rarefaction_curves(abundance_table, samples_meta, curves, feature):
    """
    Calculate rarefaction curves for all samples in the pivot abundance_table.
    Groups curves by `feature`.
    """
    for sample in tqdm(abundance_table.columns, desc="Calculating rarefaction curves"):
        # Get feature value
        feature_val = samples_meta.loc[sample, feature]
        
        # Get reads for this sample
        reads = np.repeat(abundance_table.index, abundance_table[sample].values)
        
        # Calculate rarefaction curve
        depths, richness = rarefaction_curve(reads)
        curves[feature_val].append((depths, richness))
    
    return curves


def calculate_min_rarefaction_depth(abundance_df: pd.DataFrame) -> int:
    if abundance_df.empty:
        return 0

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
    if min_depth is None:
        min_depth = calculate_min_rarefaction_depth(abundance_df)

    rarefied_tables = {tax_level: rarefy_table(abundance_df, depth=min_depth)}
    
    return rarefied_tables


def get_rarefaction_curves(abundance_table, samples_meta, config):
    tax_level = config['taxonomy']['analysis_level']
    feature = config['feature']
    pkl_path = Path(config['input']['cache_dir']) / f"rarefaction_curves_{tax_level}.pkl"

    # Check if already computed
    if pkl_path.exists():
        logger.info("Loading cached rarefaction curves...")
        with open(pkl_path, 'rb') as f:
            curves_per_feature = pickle.load(f)
        calc_rarefaction = False
    else:
        logger.info("Computing rarefaction curves...")
        curves_per_feature = {}
        calc_rarefaction = True
    
    if calc_rarefaction:
        curves = defaultdict(list)
        curves = calc_rarefaction_curves(abundance_table, samples_meta, curves, feature=feature)
        curves_per_feature[feature] = curves
    
    if calc_rarefaction:
        with open(pkl_path, 'wb') as f:
            pickle.dump(curves_per_feature, f)
        logger.info("Rarefaction curves cached.")
