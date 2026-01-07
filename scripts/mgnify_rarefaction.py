#!/usr/bin/env python3
"""
MGnify rarefaction script extracted from the notebooks.

- Reads config at the top of this file (and allows overriding via CLI soon).
- Saves the config JSON into the output directory for reproducibility.
- Uses helper functions from mgnify_utils.py
"""
from __future__ import annotations

import os
import sys
import json
import argparse
import pandas as pd

# When running as a module (`python -m scripts.mgnify.mgnify_rarefaction`),
# use relative imports from the package.
from mgnify_methods.utils.io import save_config
from mgnify_methods.utils.api import retrieve_summary
from mgnify_methods.taxonomy import pivot_taxonomic_data, invert_pivot_taxonomic_data, wide_to_long_with_ranks


# ---------------------------
# Configuration (edit as needed)
# ---------------------------
CONFIG = {
    "study_id": "MGYS00006680",
    "matching_string": "Taxonomic assignments SSU",
    "output_dir": "./outputs/mgnify_rarefaction",
    # Processing options
    "abundance_col": "abundance",
    "tax_id_col": "ncbi_tax_id",
    "taxonomy_ranks": [
        "superkingdom", "kingdom", "phylum", "class", "order", "family", "genus", "species"
    ],
    "drop_missing_tax_id": False,
    "fill_missing": True,
    "strict": False,
}


def parse_args():
    p = argparse.ArgumentParser(description="MGnify rarefaction pipeline")
    p.add_argument("--study-id", default=None, help="Override study id")
    p.add_argument("--output-dir", default=None, help="Override output directory")
    p.add_argument("--matching-string", default=None, help="Override download label")
    return p.parse_args()


def main():
    args = parse_args()
    print("Starting MGnify rarefaction script...")

    # Merge CLI overrides
    config = CONFIG.copy()
    if args.study_id:
        config["study_id"] = args.study_id
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.matching_string:
        config["matching_string"] = args.matching_string

    out_dir = config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    # Save effective config
    cfg_path = save_config(config, out_dir)
    print(f"Saved config to {cfg_path}")

    # 1) Download the study TSV
    print("Downloading MGnify summary TSV…")
    tsv_path = retrieve_summary(
        studyId=config["study_id"],
        matching_string=config["matching_string"],
        out_dir=out_dir,
    )
    print(f"Saved TSV to {tsv_path}")

    # 2) Load and do minimal cleaning like notebook did
    print("Loading TSV…")
    df = pd.read_csv(tsv_path, sep='\t')
    df = df.rename(columns={'#SampleID': 'taxonomy'})
    df = df.set_index('taxonomy')

    # Bring to long form with parsed ranks
    long_df = wide_to_long_with_ranks(
        df,
        taxonomy_col="taxonomy",
        abundance_col=config["abundance_col"],
        taxonomy_ranks=config["taxonomy_ranks"],
    )

    # 3) Pivot back using utility
    print("Pivoting taxonomic data…")
    # Note: This call expects rank columns; adapt as needed for your data shape.
    try:
        pivot = pivot_taxonomic_data(
            long_df,
            abundance_col=config["abundance_col"],
            tax_id_col=config["tax_id_col"],
            taxonomy_ranks=config["taxonomy_ranks"],
            drop_missing_tax_id=config["drop_missing_tax_id"],
            fill_missing=config["fill_missing"],
            strict=config["strict"],
        )
    except Exception as e:
        print("Pivot failed due to missing columns; saving long table only.")
        print(e)
        pivot = None

    # 4) Save outputs
    long_path = os.path.join(out_dir, "long_table.parquet")
    print(f"Saving long table → {long_path}")
    long_df.to_parquet(long_path, index=False)

    if pivot is not None:
        pivot_path = os.path.join(out_dir, "pivot_table.parquet")
        print(f"Saving pivot table → {pivot_path}")
        pivot.to_parquet(pivot_path)

    print("Done.")


if __name__ == "__main__":
    sys.exit(main())
