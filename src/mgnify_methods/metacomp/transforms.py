import numpy as np
import pandas as pd
from mgnify_methods.metacomp.rarefaction import (
    calculate_min_rarefaction_depth,
    rarefy_taxon
)

from mgnify_methods.utils.logging import get_logger
logger = get_logger(__name__, level="INFO")


def apply_preprocessing_method(table, method, config):
    sample_type = config['samples']['sample_type']
    if method == 'rarefaction':
        logger.info("\n=== Applying Rarefaction ===")
        if config['preprocess']['method_params']['rarefaction']['depth'] is None:
            logger.info("\n=== Calculating Minimum Rarefaction Depth ===")
            
            MIN_DEPTH = calculate_min_rarefaction_depth(table)
            config['preprocess']['method_params']['rarefaction']['depth'] = MIN_DEPTH

        preprocess_tables = {
            sample_type: rarefy_taxon(table, config),
            }
    
    elif method == 'TSS':
        logger.info("\n=== Applying Total Sum Scaling (TSS) ===")
        preprocess_tables = {
            sample_type: tss_transform(table),
        }
    elif method == 'TSS_sqrt':
        logger.info("\n=== Applying Square Root of Total Sum Scaling (TSS_sqrt) ===")
        preprocess_tables = {
            sample_type: tss_sqrt_transform(table),
        }
    else:
        raise ValueError(f'Method {method} not available')
    return preprocess_tables


def apply_transform_method(df, config):
    method = config['transform']['method']
    transform_params = config['transform'].get('params', {})
    logger.info(f"\n=== Applying transform {method} ===")
    if method == 'clr':
        pseudo_count = transform_params.get('pseudo_count', 1e-9)
        transformed = clr_transform((df + pseudo_count).values)
        return pd.DataFrame(transformed, index=df.index, columns=df.columns)
    else:
        raise ValueError(f"Transform {method} not available")

# dividing each row
def clr_transform(X):
    """
    Compute the Centered Logratio (CLR) transformation.
    """
    X = np.where(X == 0, 1e-9, X)  # Replace zeros
    gm = np.exp(np.log(X).mean(axis=1, keepdims=True))
    return np.log(X / gm)


def tss_transform(df):
    """
    Compute the Total Sum Scaling (TSS) transformation.
    """
    return df.div(df.sum(axis=1), axis=0)


def tss_sqrt_transform(df):
    """
    Compute the square root of Total Sum Scaling (TSS) transformation.
    """
    tss = tss_transform(df)
    return np.sqrt(tss)