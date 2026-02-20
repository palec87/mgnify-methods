from __future__ import annotations # Allows using types before they are defined/imported

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple
import pandas as pd
import pickle
from mgnify_methods.metacomp.diversity import permanova_stat_dfs, run_permanova_factors
import numpy as np
import gc
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from momics.utils import load_and_clean
from momics.taxonomy import (
    fill_taxonomy_placeholders,
    rarefy_table,
    clean_tax_row,
)

from mgnify_methods.utils.io import (
    process_analysis_metadata,
    process_samples_metadata,
    enhance_samples_metadata,
    filter_number_reads,
    align_emobon_metadata,
    add_meta,
    load_taxonomy_summary,
    filter_tax_summary,
    evaluate_completeness,
)
from mgnify_methods.taxonomy import (
    fill_lower_taxa,
    pivot_taxonomic_data,
    aggregate_by_taxonomic_level,
)
from mgnify_methods.utils.plot import (
    plot_feature_reads_hist,
)
from mgnify_methods.stats import extract_sample_stats_bulk
from mgnify_methods import AbundanceTable
from mgnify_methods.utils.logging import get_logger

if TYPE_CHECKING:
    import logging
    from mgnify_methods import TaxonomyTable


logger = get_logger(__name__, level="INFO")


def run_differential_pipeline(
    table: pd.DataFrame,
    samples_meta: pd.DataFrame,
    config: Dict,
    logger: logging.Logger,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    """Run differential abundance analysis with pairwise comparisons.

    Main orchestrator for differential abundance analysis using DESeq2
    with all pairwise comparisons between factor levels. Prepares inputs,
    runs DESeq2 for each contrast, and filters results.

    Args:
        table: Taxonomic abundance table with samples as columns.
        samples_meta: DataFrame containing sample metadata.
        config: Configuration dictionary containing DESeq2 parameters,
            sample type, and taxonomic level settings.
        logger: Logger instance for tracking progress.

    Returns:
        Dictionary with structure:
        {sample_type: {design_factor: {contrast_name: {'full': results_df,
                                                       'significant': filtered_df}}},
         'dds': DeseqDataSet}
    """

    design_factors = config["differential_abundance"]['params']['deseq2']["design_factors"]
    min_count = config["differential_abundance"]['params']['deseq2']["min_counts"]
    tax_level = config['taxonomy']['analysis_level']
    cpus = config["differential_abundance"]['params']['deseq2'].get("n_cpus", 2)

    logger.info(f"  Tax level: {tax_level}")

    counts, meta = prepare_deseq_inputs(
        table,
        samples_meta,
        design_factors,
        min_count,
    )

    dds = run_deseq2(
        counts,
        meta,
        design_factors,
        ncpus=cpus,
        # ref_level=config["differential_abundance"].get("reference")
    )
    return dds


def master_loading_pipeline(root_dir: Path, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load, merge, and preprocess MGnify and EMO-BON data.

    Master pipeline that loads data from multiple sources (MGnify and EMO-BON),
    merges them, filters samples based on read counts, processes taxonomy,
    and aggregates to the specified taxonomic level. Optionally saves
    preprocessed data for later use.

    Args:
        root_dir: Root directory containing data and output folders.
        config: Configuration dictionary specifying datasets, taxonomic
            settings, filtering parameters, and preprocessing options.

    Returns:
        Tuple of (abundance_table, samples_metadata) where:
            - abundance_table: DataFrame with taxa as rows, samples as columns,
              aggregated to specified taxonomic level.
            - samples_metadata: DataFrame with sample metadata including
              collection dates, seasons, and sequencing information.
    """
    analysis_meta, samples_meta = load_mgnify_meta(
        path=Path(config['input']['cache_dir']),
        datasets=config['datasets'],
    )

    # loads both ssu and emobon metadata, 
    abundance_emobon, emobon_meta = load_emobon(root_dir, ret='ssu')
    abundance_emobon = process_emobon_data(abundance_emobon, config)   # processing mean filling taxonomy placeholders, pivoting, cleaning taxonomic information, unifying column and index names
    logger.info(f"Loaded EMO-BON SSU data: {abundance_emobon.shape[0]} taxa, {abundance_emobon.shape[1]} samples")

    # merge EMO-BON metadata with MGnify metadata, aligning on sample IDs
    samples_meta = pd.concat([emobon_meta, samples_meta], axis=0, sort=False)

    # Filter samples based on number of reads
    samples_meta = reads_filtering(samples_meta, config=config)


    ###########################
    #### Abundance table load and process
    ###########################
    # load MGnigy data
    abundance_mgnify = load_taxonomy_summary(
        config['datasets'],
        root_dir / 'data',
    )

    # merge EMO-BON SSU and df_tax_summary on index
    abundance = abundance_mgnify.merge(
        abundance_emobon,
        left_index=True,
        right_index=True,
        how='outer',
    ).fillna(0)

    logger.info(f"Merged MGnify and EMO-BON data: {abundance.shape[0]} taxa, {abundance.shape[1]} samples")

    abundance, samples_meta = filter_tax_summary(abundance, samples_meta)

    del abundance_emobon
    del analysis_meta
    del emobon_meta
    gc.collect()

    # turn abundance into taxonomy for cleaning lineaages
    taxonomy_table = clean_abundance_table(abundance, config)

    # selection of prok or Euk for now, will be a filter type in the future perhaps.
    taxonomy_table = filter_sample_type(
        samples_meta=samples_meta,
        table=taxonomy_table,
        config=config,
    )

    ### Common preprocessing for real analysis starts here ###
    ##########################################################
    tax_level = config['taxonomy']['analysis_level']
    abundance_table = aggregate_by_taxonomic_level(
        taxonomy_table.df_filt,
        level=tax_level,
        dropna=config['diversity']['alpha']['dropna']
    )
    del taxonomy_table

    abundance_table = pivot_taxonomic_data(abundance_table)
    type(abundance_table)

    ## Here I need to remove rows which have undefined tax_level, basically ;p__; for example if tax_level is 'genus', I need to remove rows which have ';g__' but not ';s__' (species level is missing). This is because these undefined taxa can cause problems for the DESeq2 analysis later on, and they also don't provide useful information. I will keep rows which have defined taxonomic information at the selected level, even if they are missing information at lower levels. For example, if tax_level is 'family', I will keep rows which have ';f__' even if they are missing ';g__' and ';s__'.
    tax_prefix = config['taxonomy']['indicators'][tax_level]
    logger.info(f"Filtering out taxa with undefined {tax_level} (prefix '{tax_prefix}')")
    logger.info(f"Abundance table before filtering {tax_level}: {abundance_table.shape}")
    abundance_table = abundance_table[~abundance_table.index.str.contains(f";{tax_prefix};")]
    logger.info(f"Abundance table after filtering {tax_level}: {abundance_table.shape}")

    if config['precompute']['saving']:
        ## here save the metadata and abundance table for the paper, 
        metadata_out_path = root_dir / "outputs" / f"metadata_{config['precompute']['tag']}.csv"
        abundance_out_path = root_dir / "outputs" / f"abundance_{config['precompute']['tag']}.csv"

        # Save metadata and abundance table
        samples_meta.to_csv(metadata_out_path, index=True)
        abundance_table.to_csv(abundance_out_path, index=True)
        logger.info(f"Saved metadata to {metadata_out_path}")
        logger.info(f"Saved abundance table to {abundance_out_path}")

    return abundance_table, samples_meta


###############################
### Permanova global runner ###
###############################
def permanova_paper(
    table: pd.DataFrame,
    samples_meta: pd.DataFrame,
    config: Dict,
):
    """Run PERMANOVA analysis and save results.

    Performs PERMANOVA (Permutational Multivariate Analysis of Variance)
    on the abundance table for specified factors, generates summary
    statistics, and saves results including p-values and F-statistics.

    Args:
        table: Taxonomic abundance table with samples as columns.
        samples_meta: DataFrame containing sample metadata.
        config: Configuration dictionary containing sample type,
            PERMANOVA factors, and output settings.

    Note:
        Saves the following files to the output folder:
        - permanova_dict.pkl: Complete results dictionary
        - permanova_p.csv: P-values summary
        - permanova_f.csv: F-statistics summary
        - permanova_p_granular.csv: Detailed p-values
        - permanova_f_granular.csv: Detailed F-statistics
    """
    sample_type = config['samples']['sample_type']
    logger.info(f"=== Running PERMANOVA for sample type: {sample_type} ===")

    results = run_permanova_factors(
        table[sample_type],
        samples_meta,
        config['diversity']['beta']['permanova_factors'],
        metric='euclidean',
    )

    df_p, df_p_granular, df_f, df_f_granular = permanova_stat_dfs(results, config['diversity']['beta']['permanova_factors'])

    out_folder = Path(config['output']['out_folder'])   
    save_path = out_folder / 'permanova_dict.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(results, f)

    df_p.to_csv(out_folder / 'permanova_p.csv')

    df_f.to_csv(out_folder / 'permanova_f.csv')

    df_p_granular.to_csv(out_folder / 'permanova_p_granular.csv')

    df_f_granular.to_csv(out_folder / 'permanova_f_granular.csv')


#############################
# DESEq2 pipeline functions #
#############################
def prepare_deseq_inputs(
    count_table: pd.DataFrame,
    metadata: pd.DataFrame,
    design_factors: str,
    min_total_count: int = 10,
):
    """Prepare count matrix and metadata for DESeq2 analysis.

    Transposes the count table, aligns samples between counts and metadata,
    filters low-abundance taxa, ensures integer counts, and converts the
    design factor to categorical type.

    Args:
        count_table: Abundance table with taxa as rows and samples as columns.
        metadata: Sample metadata DataFrame indexed by sample ID.
        design_factors: Name of the metadata column to use as design factor.
        min_total_count: Minimum total count threshold across all samples
            for retaining a taxon. Defaults to 10.

    Returns:
        Tuple of (counts, metadata) where:
            - counts: DataFrame with samples as rows and taxa as columns,
              containing integer counts.
            - metadata: Aligned metadata DataFrame with design_factors as
              categorical type.
    """

    # transpose -> samples x taxa
    counts = count_table.T.copy()

    # align samples
    common = counts.index.intersection(metadata.index)
    counts = counts.loc[common]
    meta = metadata.loc[common]

    # filter low counts across dataset
    keep = counts.sum(axis=0) >= min_total_count
    counts = counts.loc[:, keep]

    # enforce integer counts
    counts = counts.round().astype(int)

    # ensure categorical factor
    meta[design_factors] = meta[design_factors].astype("category")

    return counts, meta


def run_deseq2(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    design_factors: list,
    ncpus: int = None,
) -> Tuple[Dict[str, pd.DataFrame], DeseqDataSet]:
    """Run DESeq2 analysis with all pairwise factor level comparisons.

    Fits a DESeq2 model and performs all pairwise comparisons between
    levels of the design factor. Each comparison tests differential
    abundance between two groups.

    Args:
        counts: Count matrix with samples as rows and taxa as columns.
        metadata: Sample metadata DataFrame with design factor.
        design_factors: List of metadata columns for comparison groups.
        ncpus: Number of CPUs to use for parallel computation. If None,
            defaults to 2.

    Returns:
        Tuple of (results_dict, dds) where:
            - results_dict: Dictionary mapping contrast names
              (e.g., 'groupA_vs_groupB') to DESeq2 results DataFrames.
            - dds: Fitted DeseqDataSet object.

    Raises:
        ValueError: If design_factor has fewer than 2 levels.
    """
    inference = DefaultInference(n_cpus=ncpus if ncpus is not None else 2)
    dds = DeseqDataSet(
        counts=counts,
        metadata=metadata,
        design_factors=design_factors,
        refit_cooks=True,
        inference=inference,
        quiet=False,
    )
    logger.info("Fitting DESeq2 model...")
    logger.info(f"  Design: ~ {design_factors}")
    logger.info(f"Dataset: {dds}")

    dds.deseq2()

    return dds


def deseq2_pairwise_contrasts(
    dds: DeseqDataSet,
    metadata: pd.DataFrame,
    design_factors: list,
) -> Dict[str, pd.DataFrame]:
    """Perform all pairwise contrasts for a DESeq2 dataset.

    Args:
        dds: Fitted DeseqDataSet object.
        metadata: Sample metadata DataFrame with design factor.
        design_factors: List of metadata columns for comparison groups.

    Returns:
        Dictionary mapping contrast names (e.g., 'groupA_vs_groupB') to
        DESeq2 results DataFrames.

    Raises:
        ValueError: If design_factors has fewer than 2 levels.
    """
    # Get unique levels of the design factor from metadata
    logger.info(f"Factor levels: {factor_levels}")
    logger.info(f"Design factors: {design_factors}")
    
    # Perform all pairwise comparisons
    result = {}
    for d in design_factors:
        result[d] = {}

        # check factor levels for this design factor
        factor_levels = sorted(metadata[d].unique())
        if len(factor_levels) < 2:
            logger.warning(f"Design factor '{d}' has less than 2 levels for contrast")
            continue

        for i, group_to_compare in enumerate(factor_levels):
            for baseline in factor_levels[:i]:
                contrast_name = f"{group_to_compare}_vs_{baseline}"
                logger.info(f"Computing contrast: {contrast_name}")
                stats = DeseqStats(dds, contrast=(d, group_to_compare, baseline))
                stats.summary()
                
                res = stats.results_df.copy()
                res.index.name = "taxon"
                result[d][contrast_name] = res

    return result


def filter_deseq_results(
    results: pd.DataFrame,
    padj_thresh: float = 0.05,
    lfc_thresh: float = None,
) -> pd.DataFrame:
    """Filter DESeq2 results by significance thresholds.

    Applies false discovery rate (FDR) filtering and optional log2 fold
    change filtering to identify significantly differentially abundant taxa.

    Args:
        results: DESeq2 results DataFrame with 'padj' and 'log2FoldChange'
            columns.
        padj_thresh: Maximum adjusted p-value threshold for significance.
            Defaults to 0.05.
        lfc_thresh: Minimum absolute log2 fold change threshold. If None,
            no fold change filtering is applied. Defaults to None.

    Returns:
        Filtered DataFrame containing only significant results, sorted by
        adjusted p-value in ascending order.
    """

    sig = results.dropna(subset=["padj"])
    sig = sig[sig["padj"] < padj_thresh]

    if lfc_thresh is not None:
        sig = sig[np.abs(sig["log2FoldChange"]) >= lfc_thresh]

    return sig.sort_values("padj")


def analyse_deseq_results(
    dds,
    meta: pd.DataFrame,
    config: Dict,
) -> Dict:
    """Perform analysis and visualization of DESeq2 results.

    Analyzes the DESeq2 results by generating summary statistics, creating
    visualizations such as volcano plots and heatmaps, and saving outputs
    to the specified output folder.

    Args:
        deseq_results: Nested dictionary containing DESeq2 results for each
            contrast.
        config: Configuration dictionary containing output settings and
            visualization parameters.

    Returns:
        Dictionary containing analysis outputs such as summary statistics,
        generated plots, and filtered significant results.
    """
    design_factors = config["differential_abundance"]['params']['deseq2']["design_factors"]
    padj = config["differential_abundance"]['params']['deseq2']["padj"]
    lfc = config["differential_abundance"]['params']['deseq2']["lfc"]
    sample_type = config['samples']['sample_type']

    result = {}

    logger.info(f"=== Running DESeq2 result analysis ===")
    logger.info(f"Sample type: {sample_type}")
    logger.info(f"Design factors: {design_factors}")
    logger.info(f"Padj threshold: {padj}, LFC threshold: {lfc}")

    result[sample_type] = {}

    res = deseq2_pairwise_contrasts(dds, meta, design_factors)

    for design_factor, contrasts in res.items():
        logger.info(f"Design factor: {design_factor}, contrasts: {list(contrasts.keys())}")
        
        result[sample_type][design_factor] = {}
        for contrast, r in contrasts.items():
            sig = filter_deseq_results(r, padj, lfc)

            result[sample_type][design_factor][contrast] = {
                "full": r,
                "significant": sig,
            }

            logger.info(
                f"    Significant taxa: {len(sig)}"
            )
    return result


#############################
### Preprocessing helpers ###
#############################
def filter_sample_type(
    samples_meta: pd.DataFrame,
    table: TaxonomyTable,
    config: Dict,
) -> TaxonomyTable:
    """Filter taxonomy table to specified sample type.

    Filters the taxonomy table to include only samples matching the
    specified sample type (e.g., 'prok' or 'euk') from the configuration.

    Args:
        samples_meta: DataFrame containing sample metadata with
            'sample_type' column.
        table: TaxonomyTable object containing taxonomy data in df attribute.
        config: Configuration dictionary with 'samples.sample_type' key
            specifying the desired sample type.

    Returns:
        Updated TaxonomyTable object with df_filt attribute containing
        only samples of the specified type.
    """
    logger.info(f"Initial samples metadata: {samples_meta.shape[0]} samples")
    # Get sample list for selected sample type
    sample_type = config['samples']['sample_type']
    sample_list = samples_meta.query("sample_type == @sample_type").index.tolist()

    # Filter taxonomy table
    table.df_filt = table.df[table.df.index.isin(sample_list)]

    logger.info(f"Filtered to {sample_type} samples: {len(sample_list)} samples")
    logger.info(f"Filtered taxonomy table: {table.df_filt.shape}")
    return table


def clean_abundance_table(table: pd.DataFrame, config: Dict) -> TaxonomyTable:
    """Convert and clean abundance table to taxonomy table format.

    Converts raw abundance table to TaxonomyTable format, fills missing
    taxonomic information, and applies taxonomy placeholder filling.

    Args:
        table: Raw abundance DataFrame indexed by taxonomic identifiers.
        config: Configuration dictionary containing 'taxonomy.ranks' for
            taxonomic level definitions.

    Returns:
        TaxonomyTable object with cleaned and filled taxonomic information,
        indexed by 'source material ID'.
    """
    # Convert abundance table to taxonomy table
    abundance_table = AbundanceTable(table.reset_index(), source='tax_mgnify_raw')
    ncbi_flag, taxonomy_table = abundance_table.to_taxonomy_table()

    taxonomy_table.df.sort_values(by=['source material ID', 'abundance'], inplace=True)

    # Fill missing taxonomic information
    taxonomy_table.df = fill_lower_taxa(taxonomy_table.df, config['taxonomy']['ranks'])
    taxonomy_table.df = fill_taxonomy_placeholders(taxonomy_table.df, config['taxonomy']['ranks'])

    if not taxonomy_table.df.index.name == 'source material ID':
        taxonomy_table.df = taxonomy_table.df.set_index('source material ID')

    logger.info(f"Taxonomy table ready: {taxonomy_table.df.shape}")
    return taxonomy_table


def process_emobon_data(table: pd.DataFrame, CONFIG: Dict) -> pd.DataFrame:
    """Process EMO-BON data to match MGnify format.

    Fills taxonomy placeholders, pivots data, cleans taxonomic rows,
    and unifies column/index names to match MGnify conventions.

    Args:
        table: Raw EMO-BON abundance DataFrame.
        CONFIG: Configuration dictionary containing 'taxonomy.ranks' for
            taxonomic level definitions.

    Returns:
        Processed DataFrame with cleaned taxonomy indexed by '#SampleID',
        compatible with MGnify data format.
    """
    ssu_filt = fill_taxonomy_placeholders(table, CONFIG['taxonomy']['ranks'])
    ssu_filt = pivot_taxonomic_data(ssu_filt)

    # unify taxonomic information to the MGnify
    ssu_filt = ssu_filt.reset_index()
    ssu_filt['taxonomic_concat'] = ssu_filt['taxonomic_concat'].apply(clean_tax_row)
    ssu_filt = ssu_filt.reset_index()
    # unify column and index names
    ssu_filt.drop(columns=['ncbi_tax_id', 'index'], inplace=True)
    ssu_filt = ssu_filt.rename(columns={
        'taxonomic_concat': '#SampleID',
    })
    ssu_filt.set_index('#SampleID', inplace=True)
    return ssu_filt


def load_emobon(root_folder: Path, ret: str = 'ssu') -> pd.DataFrame:
    """Load EMO-BON data and metadata.

    Loads EMO-BON sample validation data, metadata, and abundance tables
    from parquet files. Adds study tags and sample type annotations.

    Args:
        root_folder: Root directory containing EMO-BON data files.
        ret: Which abundance type to return ('ssu' for SSU rRNA data).
            Defaults to 'ssu'.

    Returns:
        Tuple of (abundance_df, metadata_df) where:
            - abundance_df: Selected abundance table (e.g., SSU).
            - metadata_df: EMO-BON metadata with 'study_tag', 'sample_type',
              and standardized column names.
    """

    def get_valid_samples():
        df_valid = pd.read_csv(
            root_folder / 'data/shipment_b1b2_181.csv'
        )
        return df_valid

    valid_samples = get_valid_samples()

    # High level function from the momics.utils module
    emobon_meta, mgf_parquet_dfs = load_and_clean(valid_samples=valid_samples)
    emobon_meta['study_tag'] = 'EMO-BON'
    emobon_meta['sample_type'] = 'prok'
    emobon_meta = align_emobon_metadata(emobon_meta)

    return mgf_parquet_dfs[ret], emobon_meta


def config_setup(root_dir: Path, config_path: Path) -> Dict:
    """Set up analysis configuration and directory structure.

    Loads configuration from JSON file, creates output and cache directories,
    optionally adds timestamps to output folder names, saves configuration
    to output folder, and configures file logging.

    Args:
        root_dir: Root directory for the project.
        config_path: Path to the JSON configuration file.

    Returns:
        Configuration dictionary with added 'output.out_folder' and
        'input.cache_dir' paths.

    Note:
        Creates the following directories if they don't exist:
        - Output folder (timestamped or 'analysis_latest')
        - Analysis cache folder
        Also saves config and sets up logging to analysis.log in output folder.
    """
    
    with open(config_path, "r") as f:  # load config from the json file
        config = json.load(f)

    # Create output directories
    if config['output']['use_timestamp']:
        stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
        out_folder = root_dir / "outputs" / ('analysis_' + stamp)
    else:
        out_folder = root_dir / "outputs" / 'analysis_latest'

    analysis_cache = root_dir / config['output']['cache_dir']

    # add out_folder to config for later use
    config['output']['out_folder'] = str(out_folder)
    config['input'] = {'cache_dir': str(analysis_cache)}

    for folder in [out_folder, analysis_cache]:
        folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directory ready: {folder}")

    # Save configuration to output folder
    config_path = out_folder / 'analysis_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to: {config_path}")

    # Configure logger to write to file in output folder
    from mgnify_methods.utils.logging import Logger
    log_file = out_folder / 'analysis.log'
    Logger.configure(level="INFO", log_file=str(log_file))
    logger.info(f"Logger configured to write to {log_file}")

    return config


def load_mgnify_meta(path: Path, datasets: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and process MGnify analysis and sample metadata.

    Loads analysis and sample metadata from MGnify, filters to pipeline
    version 5.0, enhances sample metadata with collection dates and seasons,
    merges analysis metadata, and extracts sample statistics.

    Args:
        path: Directory path for cached metadata files.
        datasets: Dictionary mapping study tags to [analysisId, ...] lists.

    Returns:
        Tuple of (analysis_meta, samples_meta) where:
            - analysis_meta: DataFrame with MGnify analysis metadata filtered
              to pipeline version 5.0.
            - samples_meta: Enhanced DataFrame with sample metadata indexed
              by 'source material ID', including sample statistics.
    """
    analysis_meta = process_analysis_metadata(path, datasets)
    analysis_meta = analysis_meta.query("`attributes.pipeline-version` == 5.0")
    logger.info(f"Loaded {len(analysis_meta)} analyses")

    # Fetch and enhance sample metadata
    samples_meta = process_samples_metadata(path, datasets)
    samples_meta = enhance_samples_metadata(samples_meta)

    samples_meta = add_meta(samples_meta, analysis_meta)
    samples_meta.set_index('relationships.run.data.id', inplace=True)
    samples_meta.index.name = 'source material ID'

    samples_meta = extract_sample_stats_bulk(samples_meta)
    logger.info(f"Loaded {len(samples_meta)} samples")

    evaluate_completeness(samples_meta)

    return analysis_meta, samples_meta


def reads_filtering(samples_meta: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Filter samples based on read count threshold.

    Optionally plots read count histograms before and after filtering,
    then removes samples with read counts below the configured threshold.

    Args:
        samples_meta: DataFrame containing sample metadata with
            'total_seq_submitted' column.
        config: Configuration dictionary containing filtering parameters,
            plot settings, and feature specification.

    Returns:
        Filtered DataFrame with low-read-count samples removed.

    Note:
        If config['plots']['reads_histogram'] is True, saves histogram
        plots showing read distribution before and after filtering.
    """
    # Plot reads distribution before filtering
    if config['plots']['reads_histogram']:
        logger.info("=== Reads Distribution (Unfiltered) ===")
        plot_feature_reads_hist(
            samples_meta=samples_meta,
            feature=config['feature'],
            name=f'{config["feature"]}_reads_unfiltered.png' if config['plots']['save_figures'] else None,
            out_dir=config['output']['out_folder'] if config['plots']['save_figures'] else None,
            use_robust_save=False,
            bins=100,
        )

    # Apply filtering
    cutoff = config['filtering']['min_reads_cutoff']
    to_drop = filter_number_reads(samples_meta, cutoff)
    logger.info(f"Filtering samples with < {cutoff} reads: {len(to_drop)} samples to remove")

    for sample in to_drop:
        samples_meta = samples_meta[samples_meta.index != sample]

    logger.info(f"Remaining analyses after filtering: {len(samples_meta)}")

    # Plot reads distribution after filtering
    if config['plots']['reads_histogram']:
        logger.info("=== Reads Distribution (Filtered) ===")
        plot_feature_reads_hist(
            samples_meta=samples_meta,
            feature=config['feature'],
            name=f'{config["feature"]}_reads_filtered.png' if config['plots']['save_figures'] else None,
            out_dir=config['output']['out_folder'] if config['plots']['save_figures'] else None,
            use_robust_save=False,
            bins=100,
        )

    return samples_meta
