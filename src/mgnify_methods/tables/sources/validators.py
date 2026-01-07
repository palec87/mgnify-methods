import pandas as pd


def validate_taxonomy_processed(df: pd.DataFrame) -> None:
    """
    Add deeper validation rules here if needed.
    Example: check numeric columns, ensure no duplicates, etc.
    """
    pass


def validate_abundance_ncbi(df: pd.DataFrame) -> None:
    """
    Placeholder for MGNify processed data validation.
    """
    if df.columns.name != "source material ID":
        raise ValueError("Columns must be named 'source material ID'")

    if set(df.index.names) != {"taxonomic_concat", "ncbi_tax_id"}:
        raise ValueError(
            "Index must be named {'taxonomic_concat', 'ncbi_tax_id'}"
        )

    if df.select_dtypes(include="number").shape[1] != df.shape[1]:
        raise ValueError("All values must be numeric")


def validate_abundance_no_ncbi(df: pd.DataFrame) -> bool:
    pass


def validate_abundance_processed(df: pd.DataFrame) -> None:
    """
    Validate MGNify processed abundance DataFrame.
    """
    # value dtypes are all numeric
    if df.select_dtypes(include='number').shape[1] != df.shape[1]:
        raise ValueError("All values must be numeric")
    
    return True
