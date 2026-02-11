import pandas as pd

from mgnify_methods.tables import TaxonomyTable, AbundanceTable


def test_taxonomy_table_to_abundance():
    df = pd.DataFrame(
        {
            "abundance": [1, 2],
            "superkingdom": ["Bacteria", "Bacteria"],
            "phylum": ["Firmicutes", "Proteobacteria"],
        },
        index=pd.Index(["S1", "S2"], name="source material ID"),
    )

    table = TaxonomyTable(df, source="tax_processed")
    abundance = table.to_abundance_table()

    assert isinstance(abundance, AbundanceTable)
    assert abundance.df.shape[1] == 2


def test_abundance_table_to_taxonomy_no_ncbi():
    df = pd.DataFrame(
        {"S1": [1, 0], "S2": [0, 2]},
        index=pd.Index(
            [
                "sk__Bacteria;k__Firmicutes;p__Bacillota",
                "sk__Bacteria;k__Proteobacteria;p__Proteobacteria",
            ],
            name="taxonomic_concat",
        ),
    )
    df.columns.name = "source material ID"

    table = AbundanceTable(df, source="abundance_processed")
    has_ncbi, tax_table = table.to_taxonomy_table()

    assert has_ncbi is False
    assert isinstance(tax_table, TaxonomyTable)
    assert "abundance" in tax_table.df.columns


def test_abundance_table_to_taxonomy_with_ncbi():
    idx = pd.MultiIndex.from_tuples(
        [
            ("sk__Bacteria;k__Firmicutes;p__Bacillota", 1),
            ("sk__Bacteria;k__Proteobacteria;p__Proteobacteria", 2),
        ],
        names=["taxonomic_concat", "ncbi_tax_id"],
    )
    df = pd.DataFrame({"S1": [1, 0], "S2": [0, 2]}, index=idx)
    df.columns.name = "source material ID"

    table = AbundanceTable(df, source="abundance_processed")
    has_ncbi, tax_table = table.to_taxonomy_table()

    assert has_ncbi is True
    assert isinstance(tax_table, TaxonomyTable)
