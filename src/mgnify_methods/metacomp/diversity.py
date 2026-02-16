from pathlib import Path
from typing import Iterable, Tuple, List, Dict
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from mgnify_methods.utils.plot import (
    create_alpha_diversity_plots,
    plot_beta,
)
from mgnify_methods.stats import (
    alpha_diversity_report,
    compare_alpha_diversities,
)

from mgnify_methods.utils.io import (
    extract_feature,
    save_plot,
    save_df,
)
# Beta diversity
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.ordination import pcoa
from skbio.stats.distance import permanova


from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def beta_diversity_analysis(abundance_table: pd.DataFrame, samples_meta: pd.DataFrame, config: dict):
    if not config['diversity']['beta']['enabled']:
        return
    logger.info("\n=== Beta Diversity Analysis ===")
    tax_level = config['taxonomy']['analysis_level']
    logger.info(f"Analyzing at {tax_level} level...")
    
    df_diversity_input = abundance_table.T
    
    logger.info(f"Data: {df_diversity_input.shape[0]} samples, {df_diversity_input.shape[1]} taxa")
    
    # Calculate beta diversity and PCoA
    if sorted(samples_meta.index) != sorted(df_diversity_input.index):
        raise ValueError("Index mismatch between samples_meta and abundance_table")
    logger.info(f"Metric: {config['diversity']['beta']['metric']}")
    dist = squareform(pdist(df_diversity_input.values,
                            metric=config['diversity']['beta']['metric'],
                            ),
                        )
    beta = DistanceMatrix(
        dist,
        ids=df_diversity_input.index.astype(str)
    )
    pcoa_result = pcoa(beta, method="eigh")
    explained_variance = (
        pcoa_result.proportion_explained[0],
        pcoa_result.proportion_explained[1],
    )
    
    # Merge with metadata
    pcoa_df = pd.merge(
        pcoa_result.samples,
        samples_meta,
        left_index=True,
        right_index=True,
        how="inner",
    )
    
    # Add feature information
    logger.info(f"Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    
    # Plot PCoA
    if config['plots']['beta_diversity_pcoa']:
        plot_beta(pcoa_df, explained_variance, config)


def alpha_diversity_analysis(
        abundance_table: pd.DataFrame,
        samples_meta: pd.DataFrame,
        config: dict):
    logger.info("\n=== Alpha Diversity Analysis ===")
    tax_level = config['taxonomy']['analysis_level']
    feature = config['feature']
    logger.info(f"Analyzing at {tax_level} level...")

    # Transpose for diversity calculation
    df_diversity_transposed = abundance_table.T
    df_diversity_transposed.index.name = 'sample_id'
    
    # Create factors DataFrame
    factors_df = pd.DataFrame(index=df_diversity_transposed.index)
    if feature not in samples_meta.columns:
        logger.info(f"Feature '{feature}' not found in analysis metadata. Attempting to extract from sample metadata...")
        # Add season information
        factors_df = extract_feature(factors_df, feature, samples_meta=samples_meta)
    else:
        factors_df[feature] = samples_meta.reindex(factors_df.index)[feature]
        factors_df[feature] = factors_df[feature].fillna('Unknown')

    logger.info(f"Data: {df_diversity_transposed.shape[0]} samples, {df_diversity_transposed.shape[1]} taxa")
    logger.info(f"Study distribution: {factors_df[feature].value_counts().to_dict()}")
    
    # Calculate diversity
    diversity_df, diversity_results, diversity_metrics = alpha_diversity_report(
        df_diversity_transposed,
        factors_df,
        feature=feature,
    )
    
    # setup saving
    out_dir = Path(config['output']['out_folder'])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot diversity
    if config['plots']['alpha_diversity']:
        fig_alpha = create_alpha_diversity_plots(diversity_df, tax_level, feature)
        if config['plots']['save_figures']:
            save_plot(fig_alpha, out_dir, "alpha_diversity", feature, tag=config['output'].get('alpha_tag'))
        plt.show()

    # Summary DFs
    summary_df = diversity_df.groupby(feature)[diversity_metrics].describe().round(3)
    stats_df = compare_alpha_diversities(diversity_df, diversity_metrics, feature)

    # save DFs
    save_df(summary_df, out_dir, "alpha_diversity_summary", feature, tag=config['output'].get('alpha_tag'))
    save_df(diversity_df, out_dir, "alpha_diversity", feature, tag=config['output'].get('alpha_tag'))
    save_df(stats_df, out_dir, "alpha_diversity_stats", feature, tag=config['output'].get('alpha_tag'))

    return summary_df, diversity_df, stats_df


def run_permanova(
    data: pd.DataFrame,
    metadata: pd.DataFrame,
    permanova_factor: str,
    permanova_group: List[str],
    permanova_additional_factors: List[str],
    metric: str = 'euclidean',
    permutations: int = 999,
    verbose: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Run PERMANOVA on the given data and metadata.
    Args:
        data (pd.DataFrame): DataFrame containing the abundance data.
        metadata (pd.DataFrame): DataFrame containing the metadata.
        permanova_factor (str): The factor to use for PERMANOVA.
        permanova_group (List[str]): List of groups to include in the analysis.
        permanova_additional_factors (List[str]): Additional factors to test.
        permutations (int): Number of permutations for PERMANOVA. Default is 999.
        verbose (bool): If True, print detailed output.
    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing PERMANOVA results for each factor.
    """
    # Filter metadata based on selected groups
    if permanova_factor == "All":
        filtered_metadata = metadata.copy()
    else:
        filtered_metadata = metadata[metadata[permanova_factor].isin(permanova_group)]

    # Match data and metadata samples
    abundance_matrix = data[filtered_metadata.index].T

    permanova_results = {}
    # factors_to_test = permanova_additional_factors
    for remaining_factor in permanova_additional_factors:
        factor_metadata = filtered_metadata.dropna(subset=[remaining_factor])
        combined_abundance = abundance_matrix.loc[factor_metadata.index]

        dist = squareform(pdist(combined_abundance.values,
                            metric=metric,
                            ),
                        )
        distance_matrix_obj = DistanceMatrix(
            dist,
            ids=combined_abundance.index
        )

        factor_metadata = factor_metadata.loc[
            factor_metadata.index.intersection(distance_matrix_obj.ids)
        ]

        if remaining_factor not in factor_metadata.columns:
            continue

        group_vector = factor_metadata[remaining_factor]
        if group_vector.nunique() < len(group_vector):
            if set(distance_matrix_obj.ids) == set(group_vector.index):
                permanova_result = permanova(
                    distance_matrix_obj,
                    grouping=group_vector,
                    permutations=permutations,
                )
                permanova_results[remaining_factor] = permanova_result
                if verbose:
                    logger.info(f"Factor: {remaining_factor}")
                    logger.info(
                        f"  F-statistic: {permanova_result['test statistic']:.4f}"
                    )
                    logger.info(f"  p-value: {permanova_result['p-value']:.4f}\n")
        else:
            logger.info(
                f"Skipping factor '{remaining_factor}' due to unique values in grouping vector."
            )

    return permanova_results


###############
### helpers ###
###############
def run_permanova_factors(df, metadata, factors_list, metric='euclidean'):

    samples_meta_no_na = metadata.dropna(axis=1)
    results = {}

    for factor in tqdm(factors_list):
        result = run_permanova(
            df,
            samples_meta_no_na,
            permanova_factor=factor,
            permanova_group=samples_meta_no_na[factor].unique().tolist(),  # all unique values of the factor
            permanova_additional_factors=[f for f in factors_list if f != factor],  # include all factors for stratification except the current one
            metric=metric,
            permutations=1999,  # increase permutations for more robust results
        )
        results[f"{factor}_all"] = result

        for sub_factor in samples_meta_no_na[factor].unique().tolist():
            try:
                sub_result = run_permanova(
                    df,
                    samples_meta_no_na,
                    permanova_factor=factor,
                    permanova_group=[sub_factor],
                    permanova_additional_factors=[f for f in factors_list if f != factor],  # include all factors for stratification except the current one
                    metric=metric,
                    permutations=1999,  # increase permutations for more robust results
                )
                results[f"{factor}_{sub_factor}"] = sub_result
            except Exception as e:
                logger.warning(f"Error running PERMANOVA for {factor}={sub_factor}: {e}")
                results[f"{factor}_{sub_factor}"] = None
    return results


def permanova_stat_dfs(results, factor_list):
    df_p = pd.DataFrame(columns=factor_list)
    df_p_granular = pd.DataFrame(columns=[f for f in factor_list if '_all' not in f])

    df_f = pd.DataFrame(columns=factor_list)
    df_f_granular = pd.DataFrame(columns=[f for f in factor_list if '_all' not in f])

    for key, result in results.items():
        if "_all" in key:
            for sub_key, item in result.items():
                df_p.loc[key.split("_all")[0], sub_key] = item['p-value']
                df_f.loc[key.split("_all")[0], sub_key] = item['test statistic']
        else:
            try:
                for sub_key, item in result.items():
                    df_p_granular.loc[key, sub_key] = item['p-value']
                    df_f_granular.loc[key, sub_key] = item['test statistic']
            except AttributeError:
                pass
    return df_p, df_p_granular, df_f, df_f_granular
