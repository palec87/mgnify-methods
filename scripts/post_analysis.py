"""
This script is for running post-analysis scripts on the outputs of the main paper run. 
It is not meant to be run as part of the main paper run, because it produces plots from the `paper_run.py` outputs.
The scripts invoked here are:
- alpha_diversity_heatmaps.py
- beta_diversity_heatmaps.py
- deseq2_plots.py
"""

import subprocess

FOLDER = "outputs/analysis_20260217_2009_class_full"


# subprocess.run([
#     "python",
#     "scripts/alpha_diversity_heatmaps.py",
#     f"{FOLDER}/alpha_diversity_stats_study_tag_no_processing.csv"
# ], check=True)

subprocess.run([
    "python",
    "scripts/alpha_diversity_heatmaps.py",
    f"{FOLDER}/alpha_diversity_stats_study_tag_remove_singletons.csv"
], check=True)

subprocess.run([
    "python",
    "scripts/beta_diversity_heatmaps.py",
    f"{FOLDER}/"
], check=True)


subprocess.run([
    "python",
    "scripts/deseq2_plots.py",
    f"{FOLDER}/deseq_result.pkl"
], check=True)