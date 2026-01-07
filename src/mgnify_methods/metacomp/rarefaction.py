import numpy as np
from tqdm import tqdm

def calc_rarefaction_curves(abund_table, metadata, curves):
    for sample in tqdm(abund_table.columns):
        # _, ratio = extract_sample_stats(metadata, sample)
        reads = np.repeat(abund_table.index, abund_table[sample].values)
        depths, richness = rarefaction_curve(reads)

        campaign = metadata[metadata.index==sample]['study_tag'].values[0]
        curves[campaign].append((depths, richness))
    return curves


def rarefaction_curve(reads, steps=20, replicates=10):
    depths = np.linspace(1, len(reads), steps, dtype=int)
    richness = []
    for n in depths:
        reps = []
        for _ in range(replicates):
            subsample = np.random.choice(reads, size=n, replace=False)
            reps.append(len(np.unique(subsample)))
        richness.append(np.mean(reps))
    return depths, richness