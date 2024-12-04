from bento_sc.data import CellSampleProcessor, SequentialPreprocessor, BatchCollater
import h5torch
from tqdm import tqdm
import torch


train = h5torch.Dataset(
    "../data/circ_imm.h5t",
    sample_processor=CellSampleProcessor(SequentialPreprocessor(*[]), return_zeros=True),
    in_memory=False,
    subset=("0/split", "test"),
)

dataloader = torch.utils.data.DataLoader(
    train,
    num_workers=4,
    batch_size=512,
    shuffle=False,
    pin_memory=True,
    collate_fn=BatchCollater(),
)

dict_keeper = []
for i in range(19331):
    dict_keeper.append(dict())



for batch in tqdm(dataloader):
    for i in range(19331):
        s, c = torch.unique(batch["gene_counts"][:, i], return_counts=True)
        for ss, cc in zip(s, c):
            if ss.item() not in dict_keeper[i]:
                dict_keeper[i][ss.item()] = cc.item()
            else:
                dict_keeper[i][ss.item()] += cc.item()


def mu(gene_c):
    total = 0
    running_sum = 0
    for k, v in gene_c.items():
        running_sum += k*v
        total += v

    return running_sum / total

def var(gene_c, mu):
    total = 0
    running_sum = 0
    for k, v in gene_c.items():
        running_sum += ((k - mu)**2)*v
        total += v

    return running_sum / total

mus = []
vars = []
for i in range(19331):
    gene_c = dict_keeper[i]
    mu_i = mu(gene_c)
    var_i = var(gene_c, mu_i)
    mus.append(mu_i)
    vars.append(var_i)

import matplotlib.pyplot as plt
import numpy as np
from skmisc.loess import loess

not_const = np.array(vars) > 0
x = np.log10(np.array(mus))[not_const]
y = np.log10(np.array(vars))[not_const]
model = loess(x, y, span=0.3, degree=2)
model.fit()


def safe_var(gene_c, mu, reg_var):
    total = sum(list(gene_c.values()))
    running_sum = 0
    for k, v in gene_c.items():
        val = (k - mu)/reg_var
        running_sum += (np.minimum(val, total**.5)**2)*v

    return running_sum / (total - 1)

final_var = np.zeros(19331)
for ix, i in enumerate(np.where(not_const)[0]):
    gene_c = dict_keeper[i]
    mu_i = mus[i]
    var_i = model.outputs.fitted_values[ix]
    final_var[i] = safe_var(gene_c, mu_i, var_i)

np.save("./hvg.npy", final_var)