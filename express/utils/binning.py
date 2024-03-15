import h5torch
from tqdm import tqdm
import numpy as np

f = h5torch.File("../cellxgene.h5t")

ss = np.random.choice(f["central/data"].shape[0], size=10_000, replace=True)
ss = np.unique(np.sort(ss))
counts = {}
for c in tqdm(ss):
    counts_section = f["central/data"][c : c + 100_000]
    for s, c in zip(*np.unique(counts_section, return_counts=True)):
        if s in counts:
            counts[s] += c
        else:
            counts[s] = c


counts_inarray = np.zeros((max(list(counts.keys()))).astype(int) + 1)
for k, v in counts.items():
    counts_inarray[k.astype(int)] = v


p = counts_inarray[1:] / counts_inarray.sum()


bins = []
n_bins = 500
rolling_percent = 0
start_loc = len(p) - 1
for i in range(len(p) - 1, -1, -1):
    rolling_percent += p[i]
    if rolling_percent > 1 / n_bins:
        bins.append(start_loc)
        start_loc = i
        rolling_percent = 0

bins = np.array(bins[::-1])
bins[-1] = 1e9
np.savetxt("../bins500.txt", bins)
