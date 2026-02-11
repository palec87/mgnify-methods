import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from mgnify_methods.utils import plot as plot_module


def test_plot_feature_reads_hist_kde(monkeypatch):
    samples_meta = pd.DataFrame(
        {
            "study_tag": ["A", "A", "B", "B"],
        },
        index=["S1", "S2", "S3", "S4"],
    )

    def fake_extract_sample_stats(_meta, sample):
        return {"S1": (10, 0.1), "S2": (100, 0.2), "S3": (1000, 0.3), "S4": (10000, 0.4)}[sample]

    monkeypatch.setattr(plot_module, "extract_sample_stats", fake_extract_sample_stats)
    monkeypatch.setattr(plot_module.plt, "show", lambda: None)

    result = plot_module.plot_feature_reads_hist(
        samples_meta=samples_meta,
        feature="study_tag",
        use_robust_save=False,
        bins=50,
    )

    assert set(result.keys()) == {"A", "B"}
    assert all(len(v) == 2 for v in result.values())
