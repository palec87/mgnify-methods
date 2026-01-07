"""
Test Taxonomy table methods when starting from the emobon Taxonomy raw table
"""


import os
import pandas as pd
from mgnify_methods import TaxonomyTable
from mgnify_methods.tables.sources.validators import validate_abundance_ncbi
from mgnify_methods.tables.sources.detectors import is_emobon_processed
import pytest
from momics.utils import load_and_clean


ROOT = os.path.abspath(os.path.join('./'))
print(f"ROOT: {ROOT}")

def get_valid_samples():
    df_valid = pd.read_csv(
        os.path.join(ROOT, 'data/shipment_b1b2_181.csv')
    )
    return df_valid


@pytest.fixture
def emobon_taxonomy_raw_df() -> pd.DataFrame:
    """
    Fixture to load the emobon Taxonomy raw table as a DataFrame.
    """
    valid_samples = get_valid_samples()
    full_metadata, mgf_parquet_dfs = load_and_clean(valid_samples=valid_samples)
    emobon_taxonomy_raw_df = mgf_parquet_dfs['ssu']
    return emobon_taxonomy_raw_df


def test_raw_to_processed(emobon_taxonomy_raw_df):
    """
    Test conversion from emobon Taxonomy raw DataFrame to TaxonomyTable.
    """
    taxonomy_table = TaxonomyTable(emobon_taxonomy_raw_df)
    assert is_emobon_processed(taxonomy_table.df), "DataFrame should be detected as emobon_processed format."


def test_abundance_and_back(emobon_taxonomy_raw_df):
    """
    Test conversion from emobon Taxonomy raw DataFrame to AbundanceTable and back.
    """
    taxonomy_table = TaxonomyTable(emobon_taxonomy_raw_df)
    abundance_df = taxonomy_table.to_abundance_table()
    validate_abundance_ncbi(abundance_df.df)

    # Convert back to TaxonomyTable
    flag_has_ncbi, back_taxonomy_table = abundance_df.to_taxonomy_table()
    assert flag_has_ncbi, "The abundance table should have NCBI taxonomy IDs."
    assert type(back_taxonomy_table).__name__== "TaxonomyTable", "Back conversion should yield a TaxonomyTable instance."
    assert back_taxonomy_table.df.equals(taxonomy_table.df), "DataFrames should be equal after round-trip conversion."


if __name__ == "__main__":
    pytest.main([__file__])
