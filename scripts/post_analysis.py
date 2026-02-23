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