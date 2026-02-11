import pandas as pd
import pytest

from mgnify_methods.tables.sources import converters, detectors, validators


def test_mgnify_raw_to_processed():
    df = pd.DataFrame({"#SampleID": ["sk__Bacteria"], "S1": [1]})
    result = converters.mgnify_raw_to_processed(df)

    assert result.index.name == "taxonomic_concat"
    assert result.columns.name == "source material ID"


def test_emobon_standardise():
    df = pd.DataFrame({"abundance": ["1", "2"]})
    result = converters.emobon_standardise(df)

    assert result["abundance"].dtype.kind in {"i", "u"}


def test_abundance_standardise():
    df = pd.DataFrame({"S1": ["1", "2"], "S2": ["3", "4"]})
    result = converters.abundance_standardise(df)

    assert result.dtypes.apply(lambda dt: dt.kind in {"i", "u"}).all()


def test_is_mgnify_raw():
    df = pd.DataFrame({"#SampleID": ["sk__Bacteria", "sk__Archaea"], "S1": [1, 2]})
    assert detectors.is_mgnify_raw(df) is True


def test_is_abundance_no_ncbi():
    df = pd.DataFrame({"S1": [1, 2]}, index=pd.Index(["t1", "t2"], name="taxonomic_concat"))
    df.columns.name = "source material ID"

    assert detectors.is_abundance_no_ncbi(df) is True


def test_validate_abundance_ncbi_raises():
    df = pd.DataFrame({"S1": [1, 2]}, index=pd.Index(["t1", "t2"], name="taxonomic_concat"))
    df.columns.name = "not source material ID"

    with pytest.raises(ValueError):
        validators.validate_abundance_ncbi(df)


def test_validate_abundance_no_ncbi_returns_none():
    df = pd.DataFrame({"S1": [1, 2]}, index=pd.Index(["t1", "t2"], name="taxonomic_concat"))
    df.columns.name = "source material ID"

    assert validators.validate_abundance_no_ncbi(df) is None
