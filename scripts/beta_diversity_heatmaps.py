#!/usr/bin/env python3
"""Generate heatmaps from beta diversity PERMANOVA tables."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

try:
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Missing dependency: seaborn. Install with `pip install seaborn`."
    ) from exc


@dataclass(frozen=True)
class Config:
    analysis_dir: Path = Path("outputs/analysis_20260216_1347")
    file_patterns: tuple[str, ...] = (
        "permanova_f.csv",
        "permanova_p.csv",
        "permanova_f_granular.csv",
        "permanova_p_granular.csv",
    )
    output_name_pattern: str = "heatmap_{filename}.png"


def _sanitize_token(value: str) -> str:
    return "_".join(value.strip().split())


def _plot_heatmap(
    table: pd.DataFrame, 
    title: str, 
    output_path: Path,
    annot_fmt: str = ".3g",
    cmap: str = "viridis",
) -> None:
    """Plot and save a heatmap for PERMANOVA results."""
    height = max(5.5, 1.0 * len(table.index))
    width = max(5.5, 1.0 * len(table.columns))
    
    plt.figure(figsize=(width, height))
    
    # For p-values, use reversed colormap (lower p = darker)
    if "p_value" in title.lower() or "_p" in str(output_path):
        cmap = "viridis_r"
    
    # Mask NaN values
    mask = table.isna()
    
    sns.heatmap(
        table, 
        annot=True, 
        fmt=annot_fmt, 
        cmap=cmap, 
        cbar=True,
        mask=mask,
        square=True,
        linewidths=0.5,
        linecolor='gray',
    )
    
    plt.title(title, fontsize=12, pad=10)
    plt.xlabel(table.columns.name or "")
    plt.ylabel(table.index.name or "")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _process_permanova_table(csv_path: Path) -> pd.DataFrame:
    """Load and prepare PERMANOVA table."""
    df = pd.read_csv(csv_path, index_col=0)
    
    # Replace empty strings with NaN for better visualization
    df = df.replace("", np.nan)
    
    # Convert to numeric where possible
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='ignore')
    
    return df


def main() -> None:
    config = Config()
    analysis_dir = config.analysis_dir
    
    if not analysis_dir.exists():
        raise SystemExit(f"Analysis directory not found: {analysis_dir}")
    
    print(f"Processing PERMANOVA tables in: {analysis_dir}")
    
    for pattern in config.file_patterns:
        csv_path = analysis_dir / pattern
        
        if not csv_path.exists():
            print(f"Skipping {pattern} (not found)")
            continue
        
        print(f"\nProcessing: {pattern}")
        
        # Load table
        table = _process_permanova_table(csv_path)
        
        if table.empty:
            print(f"  Empty table, skipping")
            continue
        
        # Determine title and format based on filename
        stem = csv_path.stem
        if "_f" in stem:
            title = "PERMANOVA F-statistic"
            annot_fmt = ".2f"
        elif "_p" in stem:
            title = "PERMANOVA p-value"
            annot_fmt = ".4f"
        else:
            title = stem.replace("_", " ").title()
            annot_fmt = ".3g"
        
        if "granular" in stem:
            title += " (Granular)"
        
        # Generate output filename
        filename = output_pattern = config.output_name_pattern.format(
            filename=stem
        )
        output_path = analysis_dir / filename
        
        # Plot and save
        _plot_heatmap(table, title, output_path, annot_fmt=annot_fmt)
        print(f"  Saved: {output_path.name}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
