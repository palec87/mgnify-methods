from pathlib import Path
from typing import TYPE_CHECKING
import pandas as pd
try:
    from IPython.display import display
except ImportError:  # Fallback for non-notebook environments.
    display = None
import matplotlib.pyplot as plt
import seaborn as sns

from mgnify_methods.utils.plot import (
    create_alpha_diversity_plots,
)

from mgnify_methods.stats import (
    extract_feature_dict,
    alpha_diversity_report,
    compare_alpha_diversities,
)
from mgnify_methods.taxonomy import (
    aggregate_by_taxonomic_level,
    pivot_taxonomic_data,
    prevalence_cutoff_abund,
    remove_singletons_per_sample,
)

from mgnify_methods.utils.io import (
    extract_feature,
)
# Beta diversity
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa
from mgnify_methods.tables import TaxonomyTable
    
from momics.taxonomy import (
    rarefy_table,
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def beta_diversity_analysis(taxonomy_table: TaxonomyTable, analysis_meta: pd.DataFrame, samples_meta: pd.DataFrame, config: dict):
    if not config['diversity']['beta']['enabled']:
        return
    logger.info("\n=== Beta Diversity Analysis ===")
    
    tax_level = config['taxonomy']['analysis_level']
    dropna = config['diversity']['alpha']['dropna']

    logger.info(f"Analyzing at {tax_level} level...")
    
    # Prepare data
    long_df_filt = aggregate_by_taxonomic_level(
        taxonomy_table.df_filt, level=tax_level, dropna=dropna
    )
    df_diversity_pivot = pivot_taxonomic_data(long_df_filt)
    
    # Apply rarefaction and prevalence filtering
    if config['rarefaction']['enabled']:
        df_diversity_pivot = rarefy_table(df_diversity_pivot, depth=config['rarefaction']['depth'])
        logger.info(f"Rarefied to depth: {int(config['rarefaction']['depth'])}")
    
    df_diversity_pivot = prevalence_cutoff_abund(
        df_diversity_pivot, 
        percent=config['diversity']['beta']['prevalence_cutoff'], 
        skip_columns=0
    )
    df_diversity_input = df_diversity_pivot.T
    
    logger.info(f"Data: {df_diversity_input.shape[0]} samples, {df_diversity_input.shape[1]} taxa")
    
    # Calculate beta diversity and PCoA
    assert sorted(analysis_meta.index) == sorted(df_diversity_input.index), "Index mismatch"
    
    beta = beta_diversity(metric=config['diversity']['beta']['metric'], counts=df_diversity_input)
    pcoa_result = pcoa(beta, method="eigh")
    explained_variance = (
        pcoa_result.proportion_explained[0],
        pcoa_result.proportion_explained[1],
    )
    
    # Merge with metadata
    pcoa_df = pd.merge(
        pcoa_result.samples,
        analysis_meta,
        left_index=True,
        right_index=True,
        how="inner",
    )
    
    # Add feature information
    feature = config['feature']
    features_dict = extract_feature_dict(analysis_meta, samples_meta, feature=feature)
    pcoa_df[feature] = 'Unknown'
    for feature_val, samples in features_dict.items():
        sample_subset = list(samples.keys())
        pcoa_df.loc[pcoa_df.index.isin(sample_subset), feature] = feature_val
    
    logger.info(f"Explained variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
    
    # Plot PCoA
    if config['plots']['beta_diversity_pcoa']:
        _, ax = plt.subplots(figsize=(12, 8))
        
        sns.scatterplot(
            data=pcoa_df,
            x='PC1', y='PC2',
            hue=feature,
            style='study_tag',
            s=100,
            alpha=0.7,
            ax=ax
        )
        
        ax.set_xlabel(f'PC1 ({explained_variance[0]:.2%} explained variance)')
        ax.set_ylabel(f'PC2 ({explained_variance[1]:.2%} explained variance)')
        ax.set_title(f'PCoA - {feature.capitalize()} (color) vs Study (style)')
        ax.grid(True, alpha=0.3)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if config['plots']['save_figures']:
            out_dir = Path(config['output']['out_folder'])
            plt.savefig(out_dir / f"beta_pcoa_{feature}.png", 
                dpi=config['plots']['dpi'], bbox_inches='tight')
        plt.show()


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
        factors_df[feature] = factors_df.index.map(
            lambda x: samples_meta[samples_meta.index == x][feature].iloc[0]
        )

    logger.info(f"Data: {df_diversity_transposed.shape[0]} samples, {df_diversity_transposed.shape[1]} taxa")
    logger.info(f"Study distribution: {factors_df[feature].value_counts().to_dict()}")
    
    # Calculate diversity
    diversity_df, diversity_results, diversity_metrics = alpha_diversity_report(
        df_diversity_transposed,
        factors_df,
        feature=feature,
    )
    
    # Save results
    out_dir = Path(config['output']['out_folder'])
    try:
        tag = config['output']['alpha_tag']
        diversity_path = out_dir / f"alpha_diversity_{tax_level}_{tag}.csv"
    except KeyError:
        diversity_path = out_dir / f"alpha_diversity_{tax_level}.csv"

    diversity_df.to_csv(diversity_path, index=False)
    logger.info(f"\nSaved diversity results to: {diversity_path}")
    
    # Display summary
    summary_df = diversity_df.groupby(feature)[diversity_metrics].describe().round(3)
    if display is not None:
        display(summary_df)
    else:
        print(summary_df)
    
    # Plot diversity
    if config['plots']['alpha_diversity']:
        fig_alpha = create_alpha_diversity_plots(diversity_df, tax_level, feature)
        if config['plots']['save_figures']:
            fig_alpha.savefig(
                out_dir / f"alpha_diversity_{tax_level}.png",
                dpi=config['plots']['dpi'], bbox_inches='tight'
            )
        plt.show()

    stats_df = compare_alpha_diversities(diversity_df, diversity_metrics, feature)
    if display is not None:
        display(stats_df)
    else:
        print(stats_df)

    return summary_df, diversity_df, stats_df