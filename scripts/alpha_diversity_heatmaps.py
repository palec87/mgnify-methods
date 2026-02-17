#!/usr/bin/env python3
"""Generate heatmaps from alpha diversity statistics tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

try:
    import seaborn as sns
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "Missing dependency: seaborn. Install with `pip install seaborn`."
    ) from exc


@dataclass(frozen=True)
class Config:
    csv_path: Path
    comparison_separator: str = " vs "
    value_columns: tuple[str, ...] = (
        # "RawP",
        "DeltaMedian",
        "RBC",
        "CliffDelta",
        # "KW_H",
        # "KW_P",
        # "KW_EpsilonSq",
        "AdjP",
    )
    output_name_pattern: str = "heatmap_{alpha_tag}_{metric}.png"


def _sanitize_token(value: str) -> str:
    return "_".join(value.strip().split())


def _extract_alpha_tag(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "alpha_diversity_stats_study_tag_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def _ensure_value_pattern(pattern: str) -> str:
    if "{value}" in pattern:
        return pattern
    if pattern.endswith(".png"):
        return pattern.replace(".png", "_{value}.png")
    return f"{pattern}_{{value}}"


def _prepare_dataframe(df: pd.DataFrame, separator: str) -> pd.DataFrame:
    comparison = df["Comparison"].str.split(separator, n=1, expand=True)
    df = df.copy()
    df["cmp_a"] = comparison[0].str.strip()
    df["cmp_b"] = comparison[1].str.strip()
    return df


def _plot_heatmap(table: pd.DataFrame, title: str, output_path: Path) -> None:
    height = max(5.5, 1.0 * len(table.index))
    width = max(5.5, 1.0 * len(table.columns))
    plt.figure(figsize=(width, height))
    sns.heatmap(table, annot=True, fmt=".3g", cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel(table.columns.name or "")
    plt.ylabel(table.index.name or "")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _get_all_campaigns(df: pd.DataFrame) -> list[str]:
    """Extract all unique campaigns in order of first appearance."""
    campaigns = []
    seen = set()
    for cmp in pd.concat([df["cmp_a"], df["cmp_b"]]):
        if cmp not in seen:
            campaigns.append(cmp)
            seen.add(cmp)
    return campaigns


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate heatmaps from alpha diversity statistics tables."
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the alpha diversity statistics CSV file",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = Config(csv_path=args.csv_path)
    csv_path = config.csv_path
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"Metric", "Comparison", *config.value_columns}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Missing columns: {', '.join(missing)}")

    df = _prepare_dataframe(df, config.comparison_separator)
    if df["cmp_a"].isna().any() or df["cmp_b"].isna().any():
        raise SystemExit("Comparison parsing failed for some rows.")

    output_dir = csv_path.parent
    alpha_tag = _sanitize_token(_extract_alpha_tag(csv_path))
    output_pattern = _ensure_value_pattern(config.output_name_pattern)
    all_campaigns = _get_all_campaigns(df)

    for metric in sorted(df["Metric"].unique()):
        metric_df = df[df["Metric"] == metric]
        metric_name = _sanitize_token(metric)
        for value_col in config.value_columns:
            table = metric_df.pivot_table(
                index="cmp_a",
                columns="cmp_b",
                values=value_col,
                aggfunc="first",
            )
            if table.empty:
                continue
            table = table.reindex(index=all_campaigns, columns=all_campaigns)
            value_name = _sanitize_token(value_col)
            filename = output_pattern.format(
                alpha_tag=alpha_tag,
                metric=metric_name,
                value=value_name,
            )
            output_path = output_dir / filename
            title = f"{metric} - {value_col}"
            _plot_heatmap(table, title, output_path)


if __name__ == "__main__":
    main()
