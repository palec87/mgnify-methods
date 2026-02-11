import numpy as np
import pandas as pd

from mgnify_methods.taxonomy import (
    replace_trailing_empty_with_none,
    parse_taxonomic_concat,
    wide_to_long_with_ranks,
    pivot_taxonomic_data,
    invert_pivot_taxonomic_data,
)


def test_replace_trailing_empty_with_none():
    df = pd.DataFrame(
        {
            "kingdom": ["A", ""],
            "phylum": ["", ""],
            "class": ["", "C"],
        }
    )
    result = replace_trailing_empty_with_none(df)

    assert result.loc[0, "phylum"] is None
    assert result.loc[0, "class"] is None
    assert result.loc[1, "phylum"] == ""


def test_parse_taxonomic_concat():
    value = "sk__Bacteria;k__Firmicutes;p__Bacillota;g__Bacillus"
    parsed = parse_taxonomic_concat(value)

    assert parsed["superkingdom"] == "Bacteria"
    assert parsed["kingdom"] == "Firmicutes"
    assert parsed["phylum"] == "Bacillota"
    assert parsed["genus"] == "Bacillus"


def test_wide_to_long_with_ranks():
    df_wide = pd.DataFrame(
        {
            "S1": [1, 0],
            "S2": [2, 3],
        },
        index=pd.Index(
            [
                "sk__Bacteria;k__Firmicutes;p__Bacillota",
                "sk__Bacteria;k__Proteobacteria;p__Proteobacteria",
            ],
            name="taxonomy",
        ),
    )

    long_df = wide_to_long_with_ranks(df_wide)

    assert "sample" in long_df.columns
    assert "abundance" in long_df.columns
    assert "phylum" in long_df.columns
    assert long_df["abundance"].min() >= 0


def test_pivot_and_invert_taxonomic_data():
    df_long = pd.DataFrame(
        {
            "abundance": [5, 1, 2, 3],
            "superkingdom": ["Bacteria"] * 4,
            "phylum": ["Firmicutes", "Firmicutes", "Proteobacteria", "Proteobacteria"],
        },
        index=pd.Index(["S1", "S1", "S2", "S2"], name="source material ID"),
    )

    pivot = pivot_taxonomic_data(df_long)

    assert set(pivot.columns) == {"S1", "S2"}

    long_back = invert_pivot_taxonomic_data(pivot, target_col=["taxonomic_concat"])

    assert "abundance" in long_back.columns
    assert long_back.index.names[0] == "source material ID"
    assert long_back["abundance"].min() >= 0
