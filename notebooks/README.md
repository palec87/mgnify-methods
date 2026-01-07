# MGnify data integration tools

Objectives:

1. Automate not only the summary table queries from the API, but also the metadata, such as sequencing depth and sequence QC, which are important for anumdance tables normalizations.
2. Build methods which would eventually enable to construct single SPARQL endpoint combining the other extra resources to allow for comparative microbial ecology across MGnify

## Module for MGnify

methods collected in the `utils.py` module, which should be in the future integrated into the momics package.

## Taxonomy summaries

Notebook `01_mgnify_summary.ipynb` shows how to download per study the taxonomy summary. The seq depths are TBC

## Notes (mostly internal by DP)

More functional parts of the codes are also in `WF5_MGnify` example.

01_mgnify_summary is originally cloned from `/home/david-palecek/coding/emo-bon/momics-demos/testing_NBs/harmonization/01_query_mgnify_summary.ipynb`
