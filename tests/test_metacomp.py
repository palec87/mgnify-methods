import numpy as np
import pandas as pd

from mgnify_methods.metacomp.rarefaction import (
    rarefaction_curve,
    calculate_min_rarefaction_depth,
    rarefy_taxon,
)
from mgnify_methods.metacomp.transforms import (
    clr_transform,
    tss_transform,
    tss_sqrt_transform,
    apply_preprocessing_method,
    apply_transform_method,
)
from mgnify_methods.metacomp.diversity import alpha_diversity_analysis


def test_calculate_min_rarefaction_depth_bounds():
    df = pd.DataFrame(
        {"s1": [3, 1], "s2": [2, 2]},
        index=["t1", "t2"],
    )
    depth = calculate_min_rarefaction_depth(df)

    assert isinstance(depth, int)
    assert depth > 0
    assert depth <= int(df.sum().min())


def test_rarefy_taxon_depth():
    df = pd.DataFrame(
        {"s1": [3, 1], "s2": [2, 2]},
        index=["t1", "t2"],
    )
    config = {
        "taxonomy": {"analysis_level": "phylum"},
        "preprocess": {"method_params": {"rarefaction": {"depth": 2}}},
    }

    result = rarefy_taxon(df, config)

    assert "phylum" in result
    rarefied = result["phylum"]
    assert list(rarefied.columns) == ["s1", "s2"]
    assert (rarefied.sum() == 2).all()


def test_tss_transform_rows_sum_to_one():
    df = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 2.0]},
        index=["s1", "s2"],
    )
    transformed = tss_transform(df)
    row_sums = transformed.sum(axis=1).round(6)

    assert np.allclose(row_sums.values, 1.0)


def test_tss_sqrt_transform_relationship():
    df = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 2.0]},
        index=["s1", "s2"],
    )
    tss = tss_transform(df)
    tss_sqrt = tss_sqrt_transform(df)

    assert np.all(tss_sqrt >= 0)
    assert np.allclose(tss_sqrt ** 2, tss)


def test_clr_transform_row_mean_zero():
    X = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
    clr = clr_transform(X)
    row_means = clr.mean(axis=1)

    assert clr.shape == X.shape
    assert np.allclose(row_means, 0.0)


def test_apply_preprocessing_method_tss():
    table = pd.DataFrame(
        {"a": [1.0, 2.0], "b": [1.0, 2.0]},
        index=["s1", "s2"],
    )
    config = {"samples": {"sample_type": "prok"}, "preprocess": {"method_params": {}}}

    result = apply_preprocessing_method(table, "TSS", config)

    assert "prok" in result
    row_sums = result["prok"].sum(axis=1).round(6)
    assert np.allclose(row_sums.values, 1.0)


def test_apply_transform_method_clr():
    df = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]},
        index=["s1", "s2", "s3"],
    )
    config = {"transform": {"method": "clr", "params": {"pseudo_count": 1e-9}}}

    transformed = apply_transform_method(df, config)

    assert transformed.shape == df.shape
    assert np.allclose(transformed.mean(axis=1).values, 0.0)


def test_alpha_diversity_analysis_basic(tmp_path):
    abundance_table = pd.DataFrame(
        {"RUN1": [1, 0, 3], "RUN2": [0, 2, 1]},
        index=["tax1", "tax2", "tax3"],
    )
    samples_meta = pd.DataFrame(
        {
            "study_tag": ["A", "B"],
        },
        index=["RUN1", "RUN2"],
    )
    config = {
        "taxonomy": {"analysis_level": "phylum"},
        "feature": "study_tag",
        "output": {"out_folder": str(tmp_path), "alpha_tag": "test"},
        "plots": {"alpha_diversity": False, "save_figures": False, "dpi": 100},
    }

    summary_df, diversity_df, stats_df = alpha_diversity_analysis(
        abundance_table,
        samples_meta,
        config,
    )

    assert summary_df.shape[0] == 2
    assert diversity_df.shape[0] == 2
    assert stats_df is not None
    assert (tmp_path / "alpha_diversity_phylum_test.csv").exists()
