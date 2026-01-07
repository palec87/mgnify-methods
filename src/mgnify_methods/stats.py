
import ast
import numpy as np
import pandas as pd
from scipy.stats import t as _t_dist
from collections import defaultdict
from typing import Iterable, Tuple

from skbio.diversity.alpha import shannon, simpson, chao1


def extract_sample_stats(metadata, sample):
    try:
        s_clean = metadata[metadata.index==sample]['attributes.analysis-summary'].values[0].strip().rstrip(')"')
        lst = ast.literal_eval(s_clean)
    except AttributeError:
        lst = metadata[metadata.index==sample]['attributes.analysis-summary'].values[0]
    df_tmp = pd.DataFrame(lst)
    total = int(df_tmp[df_tmp['key']=='Submitted nucleotide sequences']['value'].values[0])
    identified = (int(df_tmp[df_tmp['key']=='Predicted SSU sequences']['value'].values[0]) + 
                    int(df_tmp[df_tmp['key']=='Predicted LSU sequences']['value'].values[0]))
    ratio = (total - identified) / total
    return total, ratio


def extract_feature_dict(analysis_meta, samples_meta, feature: str = 'season'):
    """
    Compute total reads per sample grouped by season.
    Returns a nested dict: {season: {sample_id: (total_reads, ratio)}}.
    """
    total_dict = defaultdict(lambda: dict())
    not_matched = 0
    if feature not in samples_meta.columns:
        raise KeyError(f"Feature '{feature}' not found in samples metadata columns.")

    for sample in analysis_meta.index:
        try:
            data_id = analysis_meta.loc[
                analysis_meta.index == sample,
                'relationships.sample.data.id'
            ].values[0]

            feature_value = samples_meta.loc[
                samples_meta['id'] == data_id,
                feature
            ].values[0]

            total_dict[feature_value][sample] = extract_sample_stats(analysis_meta, sample)

        except IndexError:
            not_matched += 1
            continue

    print(f"Samples not matched to {feature} metadata: {not_matched}")
    return total_dict


def mean_ci_curves(
    curves: Iterable,
    n_points: int = 50,
    ci: float = 0.95,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pointwise mean and confidence interval of curves possibly with different x-grids.

    Parameters
    ----------
    curves : iterable
        Each element is a (x, y) pair of array-likes.
    n_points : int
        Number of points in auto-generated grid (if x_grid is None).
    ci : float
        Confidence level in (0,1) for the returned interval (e.g., 0.95).

    Returns
    -------
    x_common : np.ndarray
        Common x grid.
    mean_y : np.ndarray
        Pointwise mean.
    ci_lower : np.ndarray
        Lower bound of the confidence interval.
    ci_upper : np.ndarray
        Upper bound of the confidence interval.
    counts : np.ndarray
        Number of contributing curves at each x position.
    """
    # normalize input to list of (x, y) numpy arrays
    norm = []
    for c in curves:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            x = np.asarray(c[0], dtype=float)
            y = np.asarray(c[1], dtype=float)
        else:
            raise ValueError("Each curve must be a 2-tuple/list of (x, y) array-likes.")
        # ensure x ascending
        if x.size == 0 or y.size == 0:
            continue
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        norm.append((x, y))

    if len(norm) == 0:
        raise ValueError("No valid curves provided.")

    # build common x grid if not provided
    xmin = min(xy[0].min() for xy in norm)
    xmax = max(xy[0].max() for xy in norm)
    if xmax == xmin:
        x_common = np.array([xmin])
    else:
        x_common = np.linspace(xmin, xmax, n_points)

    # interpolate each curve; put NaN outside original range
    interp_vals = []
    for x, y in norm:
        y_interp = np.interp(x_common, x, y)  # fills by extrapolation; we'll mask
        mask = (x_common < x[0]) | (x_common > x[-1])
        y_interp = y_interp.astype(float)
        y_interp[mask] = np.nan
        interp_vals.append(y_interp)

    arr = np.vstack(interp_vals)  # shape (n_curves, n_x)

    mean_y = np.nanmean(arr, axis=0)
    # sample std (ddof=1) where at least 2 samples exist; else nan
    std_y = np.nanstd(arr, axis=0, ddof=1)
    counts = np.sum(~np.isnan(arr), axis=0)

    # standard error: std / sqrt(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        stderr = std_y / np.sqrt(counts)

    # degrees of freedom
    dof = counts - 1

    # get critical value for two-sided CI
    alpha = 1.0 - ci
    crit = np.full_like(mean_y, np.nan, dtype=float)
    # use t.ppf for each dof where dof >= 1
    mask_ok = dof >= 1
    crit_val = _t_dist.ppf(1 - alpha / 2.0, dof[mask_ok])
    crit[mask_ok] = crit_val

    # CI bounds: mean ± crit * stderr (where counts >=2)
    ci_half = crit * stderr
    ci_lower = mean_y - ci_half
    ci_upper = mean_y + ci_half

    # for positions with counts < 1 (no data), set everything to nan
    ci_mask = counts < 1
    mean_y[ci_mask] = np.nan
    ci_lower[ci_mask] = np.nan
    ci_upper[ci_mask] = np.nan

    # for positions with only 1 sample, stderr is NaN (std ddof=1 is NaN) -> keep NaN for CI
    single_mask = counts == 1
    ci_lower[single_mask] = np.nan
    ci_upper[single_mask] = np.nan

    return x_common, mean_y, ci_lower, ci_upper, counts


# here comes the expensive calculation part
def mean_std_curves(
    curves: Iterable,
    n_points: int = 200,
    interp_kind: str = "linear",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pointwise mean and std of an iterable of curves that may have different x grids / lengths.

    Parameters
    ----------
    curves
        Iterable of curves. Each item is a 2-tuple/list (x, y) where x and y are array-like of same length.
    n_points
        Number of points in the auto-generated common grid (used only if x_grid is None).
    interp_kind
        Only "linear" is supported in this implementation (uses numpy.interp). Kept for API clarity.

    Returns
    -------
    x_common : np.ndarray
        The common x grid.
    mean_y : np.ndarray
        Pointwise mean (ignoring NaNs).
    std_y : np.ndarray
        Pointwise standard deviation (ignoring NaNs).
    counts : np.ndarray
        Number of curves contributing (non-NaN) at each x position.
    """
    # normalize input to list of (x, y) numpy arrays
    norm = []
    for c in curves:
        if isinstance(c, (tuple, list)) and len(c) == 2:
            x = np.asarray(c[0], dtype=float)
            y = np.asarray(c[1], dtype=float)
        else:
            raise ValueError("Each curve must be a 2-tuple/list of (x, y) array-likes.")
        # ensure x ascending
        if x.size == 0 or y.size == 0:
            continue
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        norm.append((x, y))

    if len(norm) == 0:
        raise ValueError("No valid curves provided.")

    # build common x grid if not provided
    xmin = min(xy[0].min() for xy in norm)
    xmax = max(xy[0].max() for xy in norm)
    if xmax == xmin:
        x_common = np.array([xmin])
    else:
        x_common = np.linspace(xmin, xmax, n_points)

    # interpolate each curve onto x_common
    interp_vals = []
    for x, y in norm:
        # numpy.interp extrapolates; we want NaN outside original x-range:
        y_interp = np.interp(x_common, x, y, left=np.nan, right=np.nan)
        # np.interp doesn't accept left/right=np.nan in older numpy versions, so fallback:
        if np.isnan(y_interp).all():
            # try explicit masking
            y_interp = np.interp(x_common, x, y)
            mask = (x_common < x[0]) | (x_common > x[-1])
            y_interp[mask] = np.nan
        else:
            # For some numpy versions np.interp with left/right=np.nan fills with nan already.
            pass
        interp_vals.append(y_interp)

    arr = np.vstack(interp_vals)  # shape (n_curves, n_x)

    mean_y = np.nanmean(arr, axis=0)
    std_y = np.nanstd(arr, axis=0, ddof=0)
    counts = np.sum(~np.isnan(arr), axis=0)

    return x_common, mean_y, std_y, counts


def calculate_observed_otus(abundance_data):
    """Calculate number of observed OTUs/taxa."""
    abundances = np.array(abundance_data)
    return np.sum(abundances > 0)

def alpha_diversity_report(df_diversity, factors_df):
    """Calculate and report alpha diversity metrics.
    
    """
    # Calculate diversity metrics for each sample
    diversity_results = []

    for sample in df_diversity.index:
        abundance_vector = df_diversity.loc[sample].values

        # Calculate diversity metrics
        shannon_skbio = shannon(abundance_vector)
        simpson_skbio = simpson(abundance_vector)
        observed_otus = calculate_observed_otus(abundance_vector)
        chao1_skbio = chao1(abundance_vector)
        
        # Get metadata
        study_tag = factors_df.loc[sample, 'study_tag']
        season = factors_df.loc[sample, 'season']
        
        # Store results
        diversity_results.append({
            'sample_id': sample,
            'study_tag': study_tag,
            'season': season,
            'shannon': shannon_skbio,
            'simpson': simpson_skbio,
            'chao1': chao1_skbio,
            'observed_otus': observed_otus,
        })

    # Convert to DataFrame
    diversity_df = pd.DataFrame(diversity_results)
    print(f"Calculated diversity for {len(diversity_df)} samples")
    diversity_metrics = ['shannon', 'simpson', 'observed_otus', 'chao1']
    return diversity_df, diversity_results, diversity_metrics