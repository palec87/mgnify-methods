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


def _plot_volcano(
    results_df: pd.DataFrame,
    contrast_name: str,
    padj_thresh: float,
    lfc_thresh: float,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create a volcano plot from DESeq2 results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color array
    colors = np.where(
        (results_df["padj"] < padj_thresh) & (np.abs(results_df["log2FoldChange"]) >= lfc_thresh),
        "red",
        "gray",
    )
    
    ax.scatter(
        results_df["log2FoldChange"],
        -np.log10(results_df["padj"]),
        alpha=0.6,
        c=colors,
        s=30,
        edgecolors="none",
    )
    
    ax.axhline(-np.log10(padj_thresh), color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(lfc_thresh, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(-lfc_thresh, color="black", linestyle="--", linewidth=1, alpha=0.5)
    
    ax.set_xlabel("log2(Fold Change)", fontsize=12)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=12)
    ax.set_title(title or f"Volcano Plot: {contrast_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_ma(
    results_df: pd.DataFrame,
    contrast_name: str,
    padj_thresh: float,
    lfc_thresh: float,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create an MA plot (log2FC vs baseMean) from DESeq2 results."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color array
    colors = np.where(
        (results_df["padj"] < padj_thresh) & (np.abs(results_df["log2FoldChange"]) >= lfc_thresh),
        "red",
        "gray",
    )
    
    ax.scatter(
        np.log10(results_df["baseMean"] + 1),
        results_df["log2FoldChange"],
        alpha=0.6,
        c=colors,
        s=30,
        edgecolors="none",
    )
    
    ax.axhline(lfc_thresh, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(-lfc_thresh, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    
    ax.set_xlabel("log10(Base Mean)", fontsize=12)
    ax.set_ylabel("log2(Fold Change)", fontsize=12)
    ax.set_title(title or f"MA Plot: {contrast_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_pvalue_hist(
    results_df: pd.DataFrame,
    contrast_name: str,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create a p-value distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pvalues = results_df["pvalue"].dropna()
    
    ax.hist(pvalues, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_xlabel("p-value", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title or f"P-value Distribution: {contrast_name}", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_lfc_dist(
    results_df: pd.DataFrame,
    contrast_name: str,
    padj_thresh: float,
    output_path: Path,
    title: Optional[str] = None,
) -> None:
    """Create a log2 fold change distribution plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    all_lfc = results_df["log2FoldChange"].dropna()
    sig_lfc = results_df[results_df["padj"] < padj_thresh]["log2FoldChange"].dropna()
    
    ax.hist(all_lfc, bins=50, alpha=0.5, label="All taxa", edgecolor="black", color="gray")
    ax.hist(sig_lfc, bins=50, alpha=0.7, label=f"Significant (padj < {padj_thresh})", 
            edgecolor="black", color="red")
    
    ax.set_xlabel("log2(Fold Change)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title or f"log2FC Distribution: {contrast_name}", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_summary_table(
    results: dict,
    sample_type: str,
    padj_thresh: float,
    output_path: Path,
) -> None:
    """Create a summary table of significant taxa per contrast."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")
    
    summary_data = []
    for contrast, data in results[sample_type].items():
        if isinstance(data, dict) and "significant" in data:
            sig_df = data["significant"]
            full_df = data["full"]
            summary_data.append([
                contrast,
                len(full_df),
                len(sig_df),
                f"{len(sig_df) / len(full_df) * 100:.1f}%",
            ])
    
    if summary_data:
        table = ax.table(
            cellText=summary_data,
            colLabels=["Contrast", "Total Taxa", "Significant", "% Significant"],
            cellLoc="center",
            loc="center",
            colWidths=[0.3, 0.2, 0.2, 0.2],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(4):
            table[(0, i)].set_facecolor("#4472C4")
            table[(0, i)].set_text_props(weight="bold", color="white")
    
    plt.title("DESeq2 Summary Statistics", fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


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
    plots_dir = output_dir / "deseq2_plots"
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
