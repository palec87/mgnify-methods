import matplotlib
matplotlib.use("Agg")

import pandas as pd

from mgnify_methods.utils import plot as plot_module
from mgnify_methods.tables import TaxonomyTable


def test_plot_taxonomic_overlap_smoke(monkeypatch, tmp_path):
    df = pd.DataFrame(
        {
            "abundance": [1, 2],
            "superkingdom": ["Bacteria", "Bacteria"],
            "kingdom": ["Firmicutes", "Proteobacteria"],
            "phylum": ["Firmicutes", "Proteobacteria"],
        },
        index=pd.Index(["S1", "S2"], name="source material ID"),
    )
    taxonomy_table = TaxonomyTable(df, source="tax_processed")
    taxonomy_table.df_filt = taxonomy_table.df

    samples_meta = pd.DataFrame(
        {"study_tag": ["A", "B"]},
        index=["S1", "S2"],
    )
    config = {
        "taxonomy": {"analysis_level": "phylum", "filter_to_bacteria": False},
        "datasets": {"A": None, "B": None},
        "plots": {"save_figures": False, "dpi": 100},
        "output": {"out_folder": str(tmp_path)},
    }

    monkeypatch.setattr(plot_module.plt, "show", lambda: None)
    plot_module.plot_taxonomic_overlap(taxonomy_table, samples_meta, config)
