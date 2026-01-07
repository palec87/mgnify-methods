from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Iterable, Optional, Union
import logging
from .taxonomy import pivot_taxonomic_data_new

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaxonomyTable:
    """
    Handle taxonomy tables and common conversions (pivoting, aggregation, normalization).

    Expected common input layouts (examples):
      - long form: columns = ["sample", "taxon", "count", "taxonomy"] where `taxonomy` is
        a delimited string like "Kingdom;Phylum;Class;Order;Family;Genus;Species"
      - wide form: rows = taxa, columns = samples, values = counts (or vice-versa)

    Parameters
    ----------
    df : pd.DataFrame
        The underlying table of counts or taxonomy.
    taxonomy_col : Optional[str]
        Name of column that contains taxonomy strings (e.g. "taxonomy"); used for splitting.
    sample_col : Optional[str]
        Name of the sample identifier column in long form (e.g. "sample").
    taxon_col : Optional[str]
        Name of the taxon identifier column in long form (e.g. "taxon").
    """

    def __init__(
            self, df: pd.DataFrame,
            source: Optional[str] = "unknown",
            taxonomy_col: Optional[str] = None,
            sample_col: Optional[str] = None,
            taxon_col: Optional[str] = None,
            ):
        self.df = df
        self.taxonomy_col = taxonomy_col
        self.sample_col = sample_col
        self.taxon_col = taxon_col
        self.source = self._check_source(source)



    def _check_source(self, source: str) -> None:
        """Basic checks on the source data based on known source types."""
        if source not in ["emobon_processed", "emobon_raw"]:

            if self.is_emobon_processed():
                source = "emobon_processed"
                logger.info("Detected source as 'emobon_processed' based on data format.")

            elif self.is_emobon_raw():
                source = "emobon_raw"
                logger.info("Detected source as 'emobon_raw' based on data format.")

            else:
                raise ValueError(f"Unknown source type: {source}")

        elif source == "emobon_raw":
            raise NotImplementedError("Conversion from emobon_raw to standard format not implemented yet.")

        elif source == "emobon_processed":
            if self.is_emobon_processed():
                logger.info("Emobon raw data -> standardizing.")
                self.df = self.convert_emobon_raw(self.df)
        
        return source

    def is_emobon_raw(self) -> bool:
        raise NotImplementedError

    def is_emobon_processed(self) -> bool:
        if not isinstance(self.df.index, pd.MultiIndex):
            return False
        if not self.df.index.names == ["source material ID", "ncbi_tax_id"]:
            return False
        if set(self.df.columns) != {'abundance', 'superkingdom', 'kingdom', 'phylum', 'class', 'order',
                                 'family', 'genus', 'species'}:
            return False
        return True

    def to_abundance_table(self) -> pd.DataFrame:
        """Convert to AbundanceTable instance."""
        pivot = pivot_taxonomic_data_new(self.df)

        return AbundanceTable(pivot, source=self.source)


class AbundanceTable:
    """
    Handle abundance tables and common conversions (pivoting, aggregation, normalization).

    Expected input a wide form table: 
        - rows: taxa
        - columns: samples
        - values: counts (or vice-versa, but not implemented yet)

    Parameters
    ----------
    df : pd.DataFrame
        The underlying table of counts or taxonomy.
    taxonomy_col : Optional[str]
        Name of column that contains taxonomy strings (e.g. "taxonomy"); used for splitting.
    sample_col : Optional[str]
        Name of the sample identifier column in long form (e.g. "sample").
    taxon_col : Optional[str]
        Name of the taxon identifier column in long form (e.g. "taxon").
    """

    DEFAULT_RANKS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    def __init__(
            self, df: pd.DataFrame,
            source: str = "unknown",
            taxonomy_ranks: Optional[Iterable[str]] = None,
            taxonomy_col: str = None,
            index_col: Optional[str] = None,
            ):
        self.df = df
        self.taxonomy_ranks = taxonomy_ranks
        self.taxonomy_col = taxonomy_col
        self.index_col = index_col
        self.source = self._check_source(source)

        # check if df contains ncbi information
        self.has_tax_ids = self._has_tax_ids()

    # ---------------------
    # IO helpers
    # ---------------------
    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        sep: str = ",",
        source: str = "unknown",
        taxonomy_col: Optional[str] = None,
        **pd_read_csv_kwargs,
    ) -> AbundanceTable:
        """Factory classmethod to load table from CSV/TSV and return AbundanceTable instance."""
        path = Path(path)
        df = pd.read_csv(path, sep=sep, **pd_read_csv_kwargs)
        return cls(df, source, taxonomy_col=taxonomy_col)

    # ---------------------
    # Validation / conversion helpers
    # ---------------------
    def _has_tax_ids(self) -> bool:
        """Check if the dataframe index contains NCBI taxonomy IDs."""
        if isinstance(self.df.index, pd.MultiIndex):
            return "ncbi_tax_id" in self.df.index.names
        
        if not self.df.index.str.startswith("sk__").any():
            return True

        return False

    def _check_source(self, source: str) -> None:
        """Basic checks on the source data based on known source types."""
        if source not in ["mgnify", "emobon_processed", "emobon"]:
            if self.is_mgnify_raw(self.df):
                source = "mgnify"
                logger.info("Detected source as 'mgnify' based on data format.")
            else:
                logger.warning(f"Unknown source type: {source}, proceeding without conversion.")
            # elif self.is_emobon_raw(self.df):
            #     source = "emobon"
            #     logger.info("Detected source as 'emobon' based on data format.")
    
        if source == "mgnify":
            if self.is_mgnify_raw(self.df):
                logger.info("raw MGNify data -> standardizing.")
                self.df = AbundanceTable.convert_mgnify_raw(self.df)

            if self.df.index.name != 'taxonomic_concat':
                raise ValueError(f"Expected index to be 'taxonomic_concat' after MGNify conversion, got '{self.df.index.name}'.")

            if self.df.columns.duplicated().any():
                raise ValueError("Duplicate column names found in MGNify data.")
            
            if len(self.df.columns) == 0:
                raise ValueError("No sample columns found in MGNify data.")

        elif source == "emobon_processed":
            if self.is_emobon_processed(self.df):
                logger.info("Emobon processed data -> standardizing.")
                self.df = self.convert_emobon_raw(self.df)
            else:
                logger.warning("Data does not appear to be in emobon_processed format.")

        return source
    
    def is_emobon_processed(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError
    
    @staticmethod
    def convert_mgnify_raw(df: pd.DataFrame) -> pd.DataFrame:
        """Convert raw MGNify dataframe to standardized format."""
        # Rename columns
        df = df.rename(columns={"#SampleID": 'taxonomic_concat'})
        df.columns.name = "source material ID"
        df.set_index('taxonomic_concat', inplace=True)
        return df

    @staticmethod
    def is_mgnify_raw(df: pd.DataFrame) -> bool:
        try:
            required_cols = {"#SampleID"}
            missing = required_cols - set(df.columns)
            starts_with_sk = bool(df['#SampleID'].str.startswith("sk__").all())
            unique_sample_ids = df['#SampleID'].is_unique
            return len(missing) == 0 and isinstance(df.index, pd.RangeIndex) and unique_sample_ids and starts_with_sk
        except Exception as e:
            logger.error(f"Error checking MGNify raw format: {e}")
            return False

    def to_taxonomy_table(self, drop_zeros: bool = True) -> pd.DataFrame:
        """Invert pivoted taxonomic data back to long form (simplified)."""
        reset = self.df.reset_index()
        melted = reset.melt(
            id_vars=[c for c in ["source material ID", "ncbi_tax_id"] if c in reset.columns],
            value_vars=[c for c in reset.columns if c not in ["source material ID", "ncbi_tax_id"]],
            var_name='sample',
            value_name="abundance",
        )

        if drop_zeros:
            melted = melted[melted["abundance"] != 0].copy()

        try:
            melted["abundance"] = melted["abundance"].astype(int)
        except Exception:
            melted["abundance"] = pd.to_numeric(melted["abundance"], errors="coerce")

        return melted, AbundanceTable(melted)
