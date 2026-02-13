import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import gc

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
from mgnify_methods import TaxonomyTable, AbundanceTable

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def master_loading_pipeline(root_dir, config):
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


    ## here save the metadata and abundance table for the paper, 
    metadata_out_path = root_dir / "outputs" / "metadata.csv"
    abundance_out_path = root_dir / "outputs" / "abundance.csv"

    # Save metadata and abundance table
    samples_meta.to_csv(metadata_out_path, index=True)
    abundance_table.to_csv(abundance_out_path, index=True)
    logger.info(f"Saved metadata to {metadata_out_path}")
    logger.info(f"Saved abundance table to {abundance_out_path}")

    return abundance_table, samples_meta

def filter_sample_type(
    samples_meta: pd.DataFrame,
    table: TaxonomyTable,
    config: Dict,
) -> TaxonomyTable:
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
    # Convert abundance table to taxonomy table
    abundance_table = AbundanceTable(table.reset_index(), source='tax_mgnify_raw')
    ncbi_flag, taxonomy_table = abundance_table.to_taxonomy_table()

    taxonomy_table.df.sort_values(by=['source material ID', 'abundance'], inplace=True)

    # Fill missing taxonomic information
    taxonomy_table.df = fill_lower_taxa(taxonomy_table.df, config['taxonomy']['ranks'])
    taxonomy_table.df = fill_taxonomy_placeholders(taxonomy_table.df, config['taxonomy']['ranks'])

    if not taxonomy_table.df.index.name == 'source material ID':
        taxonomy_table.df = taxonomy_table.df.set_index('source material ID')

    logger.info(f"\nTaxonomy table ready: {taxonomy_table.df.shape}")
    return taxonomy_table


def process_emobon_data(table: pd.DataFrame, CONFIG: Dict) -> pd.DataFrame:
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
        print(f"Directory ready: {folder}")

    # Save configuration to output folder
    config_path = out_folder / 'analysis_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\nConfiguration saved to: {config_path}")

    # Configure logger to write to file in output folder
    from mgnify_methods.utils.logging import Logger
    log_file = out_folder / 'analysis.log'
    Logger.configure(level="INFO", log_file=str(log_file))
    logger.info(f"Logger configured to write to {log_file}")

    return config


def load_mgnify_meta(path: Path, datasets: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    # count nans in samples_meta
    nan_counts = samples_meta.isna().sum()
    logger.info(f"NaN counts in samples metadata:\n{nan_counts}")
    logger.info(f"Samples metadata shape: {samples_meta.shape}")
    return analysis_meta, samples_meta


def reads_filtering(samples_meta: pd.DataFrame, config: Dict) -> pd.DataFrame:
    # Plot reads distribution before filtering
    if config['plots']['reads_histogram']:
        logger.info("\n=== Reads Distribution (Unfiltered) ===")
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
    logger.info(f"\nFiltering samples with < {cutoff} reads: {len(to_drop)} samples to remove")

    for sample in to_drop:
        samples_meta = samples_meta[samples_meta.index != sample]

    logger.info(f"Remaining analyses after filtering: {len(samples_meta)}")

    # Plot reads distribution after filtering
    if config['plots']['reads_histogram']:
        logger.info("\n=== Reads Distribution (Filtered) ===")
        plot_feature_reads_hist(
            samples_meta=samples_meta,
            feature=config['feature'],
            name=f'{config["feature"]}_reads_filtered.png' if config['plots']['save_figures'] else None,
            out_dir=config['output']['out_folder'] if config['plots']['save_figures'] else None,
            use_robust_save=False,
            bins=100,
        )

    return samples_meta
