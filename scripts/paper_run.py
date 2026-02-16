"""
1. load config from the json file
2. load/download the tsv file from MGnify using the API
3. process metadata and taxonomic abundance data
4. Alpha diversity analysis
5. Beta diversity analysis
6. Differential abundance analysis
"""

import pandas as pd
import numpy as np
import gc
import pickle as pkl
from pathlib import Path

from mgnify_methods.taxonomy import (
    remove_singletons_per_sample,
    prevalence_cutoff_abund,
)
from mgnify_methods.metacomp.transforms import (
    apply_preprocessing_method,
    apply_transform_method,
)
from mgnify_methods.metacomp.diversity import (
    alpha_diversity_analysis,
    beta_diversity_analysis,
)

import mgnify_methods.paper_modules as pm

# Warning verbosity
import warnings 
warnings.filterwarnings(action="ignore")

from mgnify_methods.utils.logging import get_logger


logger = get_logger('paperRun', level="INFO")
ROOT_DIR = Path(__file__).parent.parent.resolve()
contig_path = ROOT_DIR / "configs" / "small.json"


CONFIG = pm.config_setup(ROOT_DIR, contig_path)

if CONFIG['precompute']['loading']:
    logger.info("Loading preprocessed data...")
    abundance_table = pd.read_csv(ROOT_DIR / "outputs" / f"abundance_{CONFIG['precompute']['tag']}.csv", index_col=0)
    samples_meta = pd.read_csv(ROOT_DIR / "outputs" / f"metadata_{CONFIG['precompute']['tag']}.csv", index_col=0)
else:
    logger.info("Running full data processing pipeline...")
    abundance_table, samples_meta = pm.master_loading_pipeline(ROOT_DIR, config=CONFIG)


################################
### Alpha diversity analysis ###
################################
if CONFIG['diversity']['alpha']['enabled']:

    # Case one - no processing, just raw data
    CONFIG['output']['alpha_tag'] = 'no_processing'
    summary_case1, df_case1, stats_case1 = alpha_diversity_analysis(
        abundance_table,
        samples_meta, CONFIG,
    )

    # Case two - remove singletons per sample
    abundance_table_alpha = abundance_table.copy()
    abundance_table_alpha = remove_singletons_per_sample(abundance_table_alpha, skip_columns=0)
    logger.info("Removed singletons per sample")

    CONFIG['output']['alpha_tag'] = 'remove_singletons'
    summary_case2, df_case2, stats_case2 = alpha_diversity_analysis(
        abundance_table_alpha,
        samples_meta, CONFIG,
    )

    # here comes saving of results, plotting, etc. for alpha diversity


###############################
### Beta diversity analysis ###
###############################
if CONFIG['diversity']['beta']['enabled']:

    ############
    ## Preprocessing for beta diversity
    ############
    if CONFIG['diversity']['beta']['remove_singletons']:
        abundance_table_beta = remove_singletons_per_sample(abundance_table, skip_columns=0)
        logger.info("Removed singletons per sample")
    else:
        abundance_table_beta = abundance_table.copy()
    
    abundance_table_beta = prevalence_cutoff_abund(
        abundance_table, 
        percent=CONFIG['diversity']['beta']['prevalence_cutoff'],
        skip_columns=0
    )

    if CONFIG['preprocess']['enabled']:
        method = CONFIG['preprocess']['method']
        logger.info(f"=== Using method {method} ===")
        
        preprocess_tables = apply_preprocessing_method(abundance_table, method, CONFIG)
    else:
        logger.info(f"No preprocessing")
        preprocess_tables = {CONFIG['samples']['sample_type']: abundance_table}

    ## Data transformation for beta diversity
    if CONFIG['transform']['enabled']:
        transformed_tables = {}
        for sample_type, table_or_dict in preprocess_tables.items():
            if isinstance(table_or_dict, dict):
                transformed_tables[sample_type] = {
                    tax_level: apply_transform_method(df, CONFIG)
                    for tax_level, df in table_or_dict.items()
                }
            else:
                transformed_tables[sample_type] = apply_transform_method(table_or_dict, CONFIG)

        preprocess_tables = transformed_tables

    ### Beta diversity analysis
    sample_type = CONFIG['samples']['sample_type']
    beta_diversity_analysis(preprocess_tables[sample_type], samples_meta, config=CONFIG)

    ### permanova analysis ###
    pm.permanova_paper(preprocess_tables, samples_meta, config=CONFIG)


#######################################
### Differential abundance analysis ###
#######################################
if CONFIG['differential_abundance']['enabled']:
    logger.info("=== Differential Abundance Analysis ===")

    deseq_results = pm.run_differential_pipeline_loop(
        abundance_table,
        samples_meta,
        CONFIG,
        logger
    )
    out_path = Path(CONFIG['output']['out_folder'])
    out_path.mkdir(parents=True, exist_ok=True)
    with open(out_path / "deseq_result.pkl", "wb") as f:
        pkl.dump(deseq_results, f)
    