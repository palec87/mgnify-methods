import matplotlib
matplotlib.use("Agg")

from pathlib import Path

from mgnify_methods.utils.plot import save_plot_with_metadata


def test_save_plot_with_metadata(tmp_path):
    result = save_plot_with_metadata(
        filename="test_plot",
        description="desc",
        plot_type="analysis",
        out_dir=tmp_path,
        save_formats=["png"],
        timestamp=False,
    )

    assert Path(result["saved_files"]["png"]).exists()
    assert Path(result["metadata_file"]).exists()
    assert Path(result["description_file"]).exists()
