import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Sequence, Tuple, Iterable
import seaborn as sns
from matplotlib import pyplot as plt
from mgnify_methods.stats import extract_sample_stats
from mgnify_methods.metacomp.rarefaction import rarefaction_curve


def plot_rarefaction_mgnify(abund_table, metadata, every_nth=20, ax=None, title="Rarefaction curves per sample"):
    if ax is None:
        fig, ax = plt.subplots()
    for sample in abund_table.columns[::every_nth]:
        _, ratio = extract_sample_stats(metadata, sample)
        reads = np.repeat(abund_table.index, abund_table[sample].values)
        depths, richness = rarefaction_curve(reads)

        ax.plot(depths, richness, label=f'{sample} (unidentified ratio: {ratio:.2f})')

    ax.legend()
    ax.set_xlabel("Number of reads")
    ax.set_ylabel("Observed richness")
    ax.set_title(title)
    return ax


def plot_season_reads_hist(
    analysis_meta,
    samples_meta,
    name=None,
    use_robust_save=True,
    **kwargs,
) -> Dict[str, Dict[str, Tuple[int, float]]]:
    total_dict = {'Spring': {}, 'Summer': {}, 'Autumn': {}, 'Winter': {}}

    # extracting the reads metadata per sample
    not_matched = 0
    for sample in analysis_meta.index:
        try:
            data_id = analysis_meta[analysis_meta.index==sample]['relationships.sample.data.id'].values[0]
            season = samples_meta[samples_meta['id']==data_id]['season'].values[0]
            total_dict[season][sample] = extract_sample_stats(analysis_meta, sample)
        except IndexError:
            not_matched += 1
            continue
    print(f"Samples not matched to season metadata: {not_matched}")
    
    # plot histogram per season 
    fig, ax = plt.subplots(figsize=(10, 6))
    season_stats = {}
    for season, stats in total_dict.items():
        totals = [stats[0] for _, stats in total_dict[season].items()]  # stat contains (total, ratio)
        ax.hist(totals, bins=10, alpha=0.3, label=season)
        season_stats[season] = {
            'n_samples': len(totals),
            'mean_reads': np.mean(totals),
            'std_reads': np.std(totals),
            'min_reads': np.min(totals),
            'max_reads': np.max(totals)
        }
    
    ax.legend()
    ax.set_xlabel("Total reads per sample")
    ax.set_ylabel("Number of samples")
    ax.set_title("Distribution of Total Reads per Sample by Season")
    
    if use_robust_save:
        # Use new robust saving method
        save_plot_with_metadata(
            fig=fig,
            filename=name.replace('.png', '') if name else "season_reads_histogram",
            description=f"Histogram showing distribution of total sequencing reads per sample, grouped by collection season. Data from MGnify study {globals().get('analysisId', 'unknown')}. Each season shows different sequencing depth patterns.",
            plot_type="histogram_seasonal",
            data_info={
                "total_samples": len(analysis_meta),
                "seasons_analyzed": list(total_dict.keys()),
                "season_statistics": season_stats,
                "study_id": globals().get('analysisId', 'unknown')
            },
            **kwargs,
        )
    
    plt.show()
    return total_dict

# -----------------------
# Plotting and saving
# -----------------------
def _json_serializer(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def _make_json_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: _make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(_make_json_serializable(item) for item in obj)
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_plot_with_metadata(
    fig=None,
    filename=None,
    description="",
    plot_type="analysis",
    data_info=None,
    save_formats=None,
    out_dir=None,
    timestamp=True,
    dpi=300,
    bbox_inches='tight',
    **kwargs
):
    """
    Robustly save plots with comprehensive metadata and description.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure, optional
        Figure to save. If None, uses plt.gcf() (current figure).
    filename : str, optional
        Base filename (without extension). If None, auto-generates from plot_type and timestamp.
    description : str
        Textual description of the plot and what it shows.
    plot_type : str
        Type of plot (e.g., "rarefaction", "diversity", "taxonomy", "comparison").
    data_info : dict, optional
        Dictionary with information about the data used (shape, samples, etc.).
    save_formats : list, optional
        List of formats to save. Default: ['png', 'pdf'].
    out_dir : str or Path, optional
        Output directory. If None, uses global OUT_FOLDER or current directory.
    timestamp : bool
        Whether to include timestamp in filename. Default: True.
    dpi : int
        Resolution for raster formats. Default: 300.
    bbox_inches : str
        Bounding box setting. Default: 'tight'.
    **kwargs
        Additional arguments passed to fig.savefig().
    
    Returns
    -------
    dict
        Dictionary with saved file paths and metadata file path.
    
    Examples
    --------
    # Basic usage
    plt.plot([1, 2, 3], [1, 4, 9])
    result = save_plot_with_metadata(
        filename="quadratic_example",
        description="Simple quadratic function showing x² relationship",
        plot_type="example"
    )
    
    # With data info
    fig, ax = plt.subplots()
    ax.hist(data, bins=20)
    result = save_plot_with_metadata(
        fig=fig,
        filename="data_histogram",
        description="Distribution of sample values after preprocessing",
        plot_type="histogram",
        data_info={
            "n_samples": len(data),
            "mean": np.mean(data),
            "std": np.std(data),
            "range": [np.min(data), np.max(data)]
        }
    )
    """
    
    # Handle defaults
    if fig is None:
        fig = plt.gcf()
    
    if save_formats is None:
        save_formats = ['png', 'pdf']
    
    if out_dir is None:
        out_dir = globals().get('OUT_FOLDER', '.')
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        base_name = f"{plot_type}_plot"
    else:
        base_name = filename
    
    # Add timestamp if requested
    if timestamp:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{base_name}_{time_str}"
    
    # Prepare metadata
    metadata = {
        "plot_info": {
            "filename_base": base_name,
            "description": description,
            "plot_type": plot_type,
            "created_at": datetime.now().isoformat(),
            "figure_size": fig.get_size_inches().tolist(),
            "dpi": int(dpi)
        },
        "data_info": _make_json_serializable(data_info or {}),
        "save_settings": {
            "formats": save_formats,
            "bbox_inches": bbox_inches,
            "dpi": int(dpi),
            "additional_kwargs": _make_json_serializable(kwargs)
        },
        "notebook_info": {
            "analysis_id": globals().get('analysisId', 'unknown'),
            "out_folder": str(out_dir),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        }
    }
    
    # Save plot in requested formats
    saved_files = {}
    for fmt in save_formats:
        plot_path = out_dir / f"{base_name}.{fmt}"
        fig.savefig(
            plot_path,
            format=fmt,
            dpi=dpi,
            bbox_inches=bbox_inches,
            **kwargs
        )
        saved_files[fmt] = str(plot_path)
        print(f"Saved {fmt.upper()}: {plot_path}")
    
    # Save metadata as JSON
    metadata_path = out_dir / f"{base_name}_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=_json_serializer)
    
    # Save description as text file
    desc_path = out_dir / f"{base_name}_description.txt"
    with open(desc_path, 'w', encoding='utf-8') as f:
        f.write(f"Plot Description\n")
        f.write(f"================\n\n")
        f.write(f"Filename: {base_name}\n")
        f.write(f"Type: {plot_type}\n")
        f.write(f"Created: {metadata['plot_info']['created_at']}\n\n")
        f.write(f"Description:\n{description}\n\n")
        if data_info:
            f.write(f"Data Information:\n")
            for key, value in data_info.items():
                f.write(f"  {key}: {value}\n")
    
    result = {
        "saved_files": saved_files,
        "metadata_file": str(metadata_path),
        "description_file": str(desc_path),
        "base_name": base_name
    }
    
    print(f"Metadata saved: {metadata_path}")
    print(f"Description saved: {desc_path}")
    
    return result


def save_current_plot(description, plot_type="analysis", **kwargs):
    """
    Convenience function to save the current matplotlib plot with description.
    
    Parameters
    ----------
    description : str
        Description of what the plot shows.
    plot_type : str
        Type of analysis or plot.
    **kwargs
        Additional arguments passed to save_plot_with_metadata.
    
    Returns
    -------
    dict
        Result from save_plot_with_metadata.
    """
    return save_plot_with_metadata(
        description=description,
        plot_type=plot_type,
        **kwargs
    )


# Updated function to integrate robust saving with existing rarefaction plots
def save_rarefaction_plot_with_metadata(fig, tax_levels, sample_type, table_shapes, description_suffix=""):
    """
    Save rarefaction plots with comprehensive metadata for taxonomic analysis.
    """
    description = f"""
    Rarefaction curves showing observed species richness versus sequencing depth across taxonomic levels.
    Analysis performed for {sample_type} samples at taxonomic levels: {', '.join(tax_levels)}.
    
    Each curve represents a different sample, with 10% prevalence filtering applied before rarefaction.
    Curves that plateau indicate sufficient sequencing depth for reliable diversity estimates.
    
    {description_suffix}
    """.strip()
    
    return save_plot_with_metadata(
        fig=fig,
        filename=f"rarefaction_{sample_type}_multilevel",
        description=description,
        plot_type="rarefaction_multilevel",
        data_info={
            "sample_type": sample_type,
            "taxonomic_levels": tax_levels,
            "table_shapes": table_shapes,
            "prevalence_cutoff": "10%",
            "analysis_type": "MGnify_taxonomic_profiling",
            "study_id": globals().get('analysisId', 'unknown')
        },
        save_formats=['png', 'pdf']
    )

# Function to save violin plots with metadata
def save_violin_plot_with_metadata(fig, plot_data, title, description):
    """Save violin/strip plots with metadata about taxonomic prevalence."""
    return save_plot_with_metadata(
        fig=fig,
        filename=f"taxonomic_prevalence_{plot_data.get('analysis_type', 'unknown')}",
        description=description,
        plot_type="violin_taxonomic_prevalence", 
        data_info=plot_data,
        save_formats=['png', 'pdf', 'svg']
    )


def plot_mean_ci(
    x: Sequence,
    mean_y: Sequence,
    ci_lower: Sequence,
    ci_upper: Sequence,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    shade_alpha: float = 0.25,
    show_counts: bool = False,
    **kwargs,
):
    """
    Plot mean curve with shaded confidence interval between ci_lower and ci_upper.

    Parameters
    ----------
    x, mean_y, ci_lower, ci_upper : sequences
        Arrays returned by mean_ci_curves.
    ax : optional
        Matplotlib axes to draw into.
    label, color : optional
        Plot label and color.
    shade_alpha : float
        Alpha for the shaded CI band.
    show_counts : bool
        If True, annotate the plot with the number of curves contributing at several x positions.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    x = np.asarray(x)
    mean_y = np.asarray(mean_y)
    ci_lower = np.asarray(ci_lower)
    ci_upper = np.asarray(ci_upper)

    ln, = ax.plot(x, mean_y, label=label, color=color)
    line_color = ln.get_color()

    ax.fill_between(x, ci_lower, ci_upper, alpha=shade_alpha, color=color or line_color)

    if label is not None:
        ax.legend()

    ax.set_xlabel(kwargs.get("xlabel", "counts"))
    ax.set_ylabel(kwargs.get("ylabel", "depth"))
    ax.set_title(kwargs.get("title", ""))

    return ax


def plot_mean_std(
    x: Sequence,
    mean_y: Sequence,
    std_y: Sequence,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    color: Optional[str] = None,
    shade_alpha: float = 0.3,
    show_points: bool = False,
    x_points: Optional[Sequence] = None,
    points: Optional[Sequence] = None,
    **kwargs,
):
    """
    Plot mean curve with shaded ± std band.

    Parameters
    ----------
    x : sequence
        x-axis positions (common grid from mean_std_curves).
    mean_y : sequence
        Mean values at x.
    std_y : sequence
        Standard deviation at x.
    ax : matplotlib Axes, optional
        Axis to plot on. If None, creates a new figure/axis.
    label : str, optional
        Label for the mean line.
    color : str, optional
        Color for the mean line and shade.
    shade_alpha : float
        Alpha for the shaded std interval.
    show_points : bool
        If True, plot raw points provided via `x_points` and `points`.
    x_points, points : sequences, optional
        If `show_points=True`, supply matching x and y arrays (or lists) of points to overlay.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    x = np.asarray(x)
    mean_y = np.asarray(mean_y)
    std_y = np.asarray(std_y)

    # main line
    ln, = ax.plot(x, mean_y, label=label, color=color)
    line_color = ln.get_color()

    # band
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=shade_alpha, color=color or line_color)

    # optional scatter of points
    if show_points and x_points is not None and points is not None:
        ax.scatter(x_points, points, s=8, color=color or "k", alpha=0.6, zorder=5)

    ax.set_xlabel(kwargs.get("xlabel", "counts"))
    ax.set_ylabel(kwargs.get("ylabel", "depth"))
    ax.set_title(kwargs.get("title", ""))
    if label is not None:
        ax.legend()

    return ax


def create_alpha_diversity_plots(diversity_df, diversity_metrics, tax_level_for_diversity):
    # Create comprehensive alpha diversity plots with study_tag as hue
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Plot 1: Shannon diversity by study
    sns.boxplot(data=diversity_df, x='study_tag', y='shannon', ax=axes[0])
    sns.stripplot(data=diversity_df, x='study_tag', y='shannon', 
              color='black', alpha=0.6, size=4, ax=axes[0])
    axes[0].set_title(f'Shannon Diversity by Study\n(Taxonomic level: {tax_level_for_diversity})')
    axes[0].set_ylabel('Shannon Index')
    axes[0].set_xlabel('Study')

    # Plot 2: Simpson diversity by study
    sns.boxplot(data=diversity_df, x='study_tag', y='simpson', ax=axes[1])
    sns.stripplot(data=diversity_df, x='study_tag', y='simpson', 
              color='black', alpha=0.6, size=4, ax=axes[1])
    axes[1].set_title(f'Simpson Diversity by Study\n(Taxonomic level: {tax_level_for_diversity})')
    axes[1].set_ylabel('Simpson Index')
    axes[1].set_xlabel('Study')

    # Plot 3: Observed OTUs by study
    sns.boxplot(data=diversity_df, x='study_tag', y='observed_otus', ax=axes[2])
    sns.stripplot(data=diversity_df, x='study_tag', y='observed_otus', 
              color='black', alpha=0.6, size=4, ax=axes[2])
    axes[2].set_title(f'Observed Taxa by Study\n(Taxonomic level: {tax_level_for_diversity})')
    axes[2].set_ylabel('Number of Observed Taxa')
    axes[2].set_xlabel('Study')

    # Plot 4: Shannon vs Simpson colored by study_tag
    sns.scatterplot(data=diversity_df, x='shannon', y='simpson', 
                hue='study_tag', s=60, alpha=0.7, ax=axes[3])
    axes[3].set_title('Shannon vs Simpson Diversity by Study')
    axes[3].set_xlabel('Shannon Index')
    axes[3].set_ylabel('Simpson Index')


    # Plot 5: Chao1 diversity by study
    sns.boxplot(data=diversity_df, x='study_tag', y='chao1', ax=axes[4])
    sns.stripplot(
        data=diversity_df, x='study_tag', y='chao1', 
        color='black', alpha=0.6, size=4, ax=axes[4]
    )
    axes[4].set_title(f'Chao1 Diversity by Study\n(Taxonomic level: {tax_level_for_diversity})')

    plt.tight_layout()
    plt.show()

    # Statistical comparison between studies
    from scipy.stats import mannwhitneyu, kruskal

    print("\n" + "="*60)
    print("STATISTICAL COMPARISON OF ALPHA DIVERSITY BETWEEN STUDIES")
    print("="*60)

    studies = diversity_df['study_tag'].unique()
    if len(studies) == 2:
        for metric in diversity_metrics:
            study1_data = diversity_df[diversity_df['study_tag'] == studies[0]][metric]
            study2_data = diversity_df[diversity_df['study_tag'] == studies[1]][metric]
        
            statistic, p_value = mannwhitneyu(study1_data, study2_data, alternative='two-sided')
        
            print(f"\n{metric.upper()} - Mann-Whitney U test:")
            print(f"  {studies[0]} (n={len(study1_data)}): median = {study1_data.median():.3f}")
            print(f"  {studies[1]} (n={len(study2_data)}): median = {study2_data.median():.3f}")
            print(f"  U-statistic = {statistic:.3f}, p-value = {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    elif len(studies) > 2:
        for metric in diversity_metrics:
            study_groups = [diversity_df[diversity_df['study_tag'] == study][metric] for study in studies]
            statistic, p_value = kruskal(*study_groups)
        
            print(f"\n{metric.upper()} - Kruskal-Wallis test:")
            for study in studies:
                study_data = diversity_df[diversity_df['study_tag'] == study][metric]
                print(f"  {study} (n={len(study_data)}): median = {study_data.median():.3f}")
            print(f"  H-statistic = {statistic:.3f}, p-value = {p_value:.4f}")
            print(f"  Significant difference: {'Yes' if p_value < 0.05 else 'No'}")

    return fig