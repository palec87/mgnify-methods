# Documenting the workflow

Separate analysis scripts are called from the `paper_run.py` which should be eventually the main runner.

## `paper_run` script

User should setup the correct config file name, which has to be located at the `repo_root/cofigs/` directory.

`pm` stands for `mgnify_methods.paper_modules` module.

If precomputed data are defined in the config, the abundance table is loaded from `repo_root/outputs/abundance_[something config].csv` together with the metadata in `repo_root/outputs/metadata_[some config].csv`. If not available `master_loading_pipeline` is invoked from `pm` module.

TODO: refactor the name paper_modules

### Alpha diversity analysis

Both `no_processing` ad `remove_singletons` are run and saved with the respective tags and stats in `alpha_diversity_analysis()`

Plotting is missing and implemented in the `post_analysis` (TODO: crosscheck this)

### Beta diversity analysis

First steps are pre-processing of the abundance table, which is the raw abundance_table loaded or calculated (nat affected by the alpha analysis).

1. Conditional removal of singletons (`mgnify_methods.taxonomy.remove_singletons_per_sample`)
2. Always filter on prevalence cutoff (`mgnify_methods.taxonomy.prevalence_cutoff_abund`)
3. Conditional preprocessing steps, only valid method currently is `rarefaction`
4. Conditional transformation of the abundance tables. This rewrites the `preprocess_tables`

Finally, the analysis is performed with `mgnify_methods.metacomp.diversity.beta_diversity_analysis`, followed always by `pm.permanova_paper`. These both have on input the `preprocess_tables`.

### Differential abundance analysis

This is independent of the previous steps, since the input is again `abundance_table`. The analysis is run by `pm.run_differential_pipeline()`, results are saved in the `dss.pkl` and `samples_meta.pkl`. Both of those serve as input to the analysis of the results by `pm.analyse_deseq_results()`, which are saved to `deseq_results.pkl`.

## `post_analysis` script

From the outputs of the `paper_run.py` (this should contain the expensive compute), plots and additional analysis scripts are invoked here, namely:

- `alpha_diversity_heatmaps.py` to generate heatmap for each metric defined in the Config dataclass inside this script. Metrics are:
    - chao, observed_OTUs, shannon, simpson.
- `beta_diversity_heatmaps.py` to generate permanova heatmaps looking for the following patterns
    - "permanova_f.csv", "permanova_p.csv", "permanova_f_granular.csv", "permanova_p_granular.csv",
- `deseq2_plots.py` to generate deseq2 volcano plots, which go into a separate subfolder `deseq2_plots`. These are many, because the plots work pairwaise on the studies.

One unfinished script `deseq2_stats` should produce some additional volcano plots.

## `ad_hoc_analysis` script

So far does plots a taxa prevalence barplot from the processed abundance tables.
