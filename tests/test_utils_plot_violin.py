import matplotlib
matplotlib.use("Agg")

import pandas as pd

from mgnify_methods.utils import plot as plot_module


def test_violin_plot_taxon_smoke(monkeypatch, tmp_path):
    comp_tables = {
        "prok": {
            "phylum": pd.DataFrame({"S1": [1, 0], "S2": [0, 1]}, index=["t1", "t2"])
        }
    }
    samples_meta = pd.DataFrame({"study_tag": ["A", "B"]}, index=["S1", "S2"])
    config = {
        "plots": {"taxa_prevalence_violin": True, "save_figures": False, "dpi": 100},
        "taxonomy": {"analysis_level": "phylum"},
        "feature": "study_tag",
        "samples": {"sample_type": "prok"},
        "output": {"out_folder": str(tmp_path)},
    }

    monkeypatch.setattr(plot_module.plt, "show", lambda: None)
    plot_module.violin_plot_taxon(comp_tables, samples_meta, config)
