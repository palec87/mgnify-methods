#!/usr/bin/env python3
"""Generate analysis plots from DESeq2 differential abundance results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle


@dataclass(frozen=True)
class Config:
    pickle_path: Path
    padj_threshold: float = 0.05
    lfc_threshold: float = 1.0
    dpi: int = 300


def _load_deseq_results(pickle_path: Path) -> dict:
    """Load DESeq2 results from pickle file."""
    if not pickle_path.exists():
        raise SystemExit(f"Pickle file not found: {pickle_path}")
    
    with open(pickle_path, "rb") as f:
        results = pickle.load(f)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate plots from DESeq2 results."
    )
    parser.add_argument(
        "pickle_path",
        type=Path,
        help="Path to the DESeq2 results pickle file",
    )
    return parser.parse_args()

def main() -> None:
    args = _parse_args()
    config = Config(pickle_path=args.pickle_path)
    results = _load_deseq_results(config.pickle_path)
    
    output_dir = config.pickle_path.parent
    sample_type = next(k for k in results.keys() if k != "dds")
    
    # Create plots directory
    plots_dir = output_dir / "deseq2_stats"
    plots_dir.mkdir(exist_ok=True)
    
    print(f"Loaded DESeq2 results from {config.pickle_path}")
    print(f"Sample type: {sample_type}")
    print(f"Contrasts: {list(results[sample_type].keys())}")
    
    # Generate plots per contrast
    for contrast, data in results[sample_type].items():
        if not isinstance(data, dict) or "full" not in data:
            continue
        
        results_df = data["full"]
        if results_df.empty:
            print(f"Skipping {contrast}: empty results")
            continue
        
        safe_contrast = "_".join(contrast.split())
        
        # Volcano plot
        volcano_path = plots_dir / f"volcano_{safe_contrast}.png"
        _plot_volcano(results_df, contrast, config.padj_threshold, config.lfc_threshold, volcano_path)
        print(f"Saved: {volcano_path}")
        
        # MA plot
        ma_path = plots_dir / f"ma_{safe_contrast}.png"
        _plot_ma(results_df, contrast, config.padj_threshold, config.lfc_threshold, ma_path)
        print(f"Saved: {ma_path}")
        
        # P-value histogram
        pval_path = plots_dir / f"pval_dist_{safe_contrast}.png"
        _plot_pvalue_hist(results_df, contrast, pval_path)
        print(f"Saved: {pval_path}")
        
        # log2FC distribution
        lfc_path = plots_dir / f"lfc_dist_{safe_contrast}.png"
        _plot_lfc_dist(results_df, contrast, config.padj_threshold, lfc_path)
        print(f"Saved: {lfc_path}")
    
    # Summary table
    summary_path = plots_dir / "summary_table.png"
    _plot_summary_table(results, sample_type, config.padj_threshold, summary_path)
    print(f"Saved: {summary_path}")
    
    print(f"\nAll plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()
