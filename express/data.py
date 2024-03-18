import h5torch
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from importlib.resources import files
from torch.distributions.binomial import Binomial
from torch.distributions.poisson import Poisson
from torch.distributions.normal import Normal
import yaml
from einops import rearrange


class ExpressDataModule(LightningDataModule):
    def __init__(self, config_path):
        super().__init__()
        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

    def setup(self, stage):

        preprocessing_mapper = {
            "RankCounts": RankCounts,
            "CountsPerX": CountsPerX,
            "FixedNorm": FixedNorm,
            "LogP1": LogP1,
            "Bin": Bin,
            "DuplicateCells": DuplicateCells,
            "FilterTopGenes": FilterTopGenes,
            "FilterRandomGenes": FilterRandomGenes,
            "PoissonResample": PoissonResample,
            "GaussianResample": GaussianResample,
            "BinomialSubsample": BinomialSubsample,
            "Mask": Mask,
            "MolecularCV": MolecularCV,
        }

        if "input_processing" in self.config:
            processor = []
            for prepr in self.config["input_processing"]:
                type_ = prepr.pop("type")
                processor.append(preprocessing_mapper[type_](**prepr))
        else:
            processor = []
        processor = SequentialPreprocessor(*processor)

        self.train = h5torch.Dataset(
            self.config["data_path"],
            sample_processor=CellSampleProcessor(
                processor, return_zeros=self.config["return_zeros"]
            ),
            in_memory=self.config["in_memory"],
            subset=("0/split", "train"),
        )

        self.val = h5torch.Dataset(
            self.config["data_path"],
            sample_processor=CellSampleProcessor(
                processor, return_zeros=self.config["return_zeros"]
            ),
            in_memory=self.config["in_memory"],
            subset=("0/split", "val"),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.config["n_workers"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.config["n_workers"],
            batch_size=self.config["batch_size"],
            shuffle=True,
            pin_memory=False,
            collate_fn=batch_collater,
        )


class CellSampleProcessor:
    def __init__(self, processor, return_zeros=False, n_genes=19331):
        self.processor = processor

        self.return_zeros = return_zeros
        self.n_genes = n_genes

    def __call__(self, f, sample):
        if self.return_zeros:
            gene_counts = np.zeros(self.n_genes)
            gene_counts[sample["central"][0]] = sample["central"][1]
            sample |= {"gene_counts": gene_counts, "gene_counts_true": gene_counts}
        else:
            asort = np.argsort(
                sample["central"][1] + np.random.rand(len(sample["central"][1]))
            )[::-1]
            gene_counts = sample["central"][1][asort]
            gene_index = sample["central"][0][asort].view(np.ndarray).astype(np.int64)
            sample |= {
                "gene_counts": torch.tensor(gene_counts),
                "gene_index": torch.tensor(gene_index),
                "gene_counts_true": torch.tensor(gene_counts),
            }

        _ = sample.pop("central")
        sample = self.processor(sample)
        return sample


class RankCounts:
    def __init__(self, key="gene_counts"):
        self.key = key

    def __call__(self, sample):
        sample[self.key] = torch.argsort(sample[self.key])
        return sample


class CountsPerX:
    def __init__(self, factor=10_000, key="gene_counts"):
        self.factor = factor
        self.key = key

    def __call__(self, sample):
        if sample[self.key].ndim == 2:
            total = sample[self.key].sum(1, keepdim=True)
        else:
            total = sample[self.key].sum()

        sample[self.key] = sample[self.key] / total * self.factor
        return sample


class FixedNorm:
    def __init__(self, factor=0.01, key="gene_counts"):
        self.factor = factor
        self.key = key

    def __call__(self, sample):
        sample[self.key] = sample[self.key] * self.factor
        return sample


class LogP1:
    def __init__(self, key="gene_counts"):
        self.key = key

    def __call__(self, sample):
        sample[self.key] = torch.log1p(sample[self.key])
        return sample


class Bin:
    def __init__(self, bins, key="gene_counts"):
        if isinstance(bins, str):
            bins = np.loadtxt(bins)
        self.bins = torch.tensor(bins)
        self.key = key

    def __call__(self, sample):
        sample[self.key] = torch.bucketize(sample[self.key], self.bins, right=True) - 1
        return sample

    @staticmethod
    def generate_bins(x, n):
        bins = np.quantile(x, np.linspace(0, 1, n + 1))
        bins = np.unique(bins)
        bins[-1] = bins[-1] + 1
        return bins


class DuplicateCells:
    def __init__(self, affected_keys=["gene_counts", "gene_index", "gene_counts_true"]):

        self.affected_keys = affected_keys

    def __call__(self, sample):
        for a in self.affected_keys:
            sample[a] = torch.stack([sample[a], sample[a].clone()])
        return sample


class FilterTopGenes:
    def __init__(
        self,
        number=1024,
        affected_keys=["gene_counts", "gene_index", "gene_counts_true"],
    ):
        self.n = number
        self.affected_keys = affected_keys

    def __call__(self, sample):
        for a in self.affected_keys:
            sample[a] = sample[a][..., : self.n]
        return sample


class FilterRandomGenes:
    def __init__(
        self,
        number=1024,
        affected_keys=["gene_counts", "gene_index", "gene_counts_true"],
    ):
        self.n = number
        self.affected_keys = affected_keys

    def __call__(self, sample):
        if sample[self.affected_keys[0]].ndim == 1:
            len_ = len(sample[self.affected_keys[0]])
            indices = torch.randperm(len_)[: self.n]
            for a in self.affected_keys:
                sample[a] = sample[a][indices]
        else:
            len_ = len(sample[self.affected_keys[0]][0])
            indices = torch.stack(
                [
                    torch.randperm(len_)[: self.n],
                    torch.randperm(len_)[: self.n],
                ]
            )
            for a in self.affected_keys:
                sample[a] = sample[a][torch.arange(2).unsqueeze(-1), indices]
        return sample


class PoissonResample:
    def __init__(self, key="gene_counts", clamp_1=True):
        self.key = key
        self.clamp = clamp_1

    def __call__(self, sample):
        resampled = Poisson(sample[self.key]).sample()
        if self.clamp:
            resampled = torch.clamp(resampled, min=1)
        sample[self.key] = resampled
        return sample


class GaussianResample:
    def __init__(self, key="gene_counts", std=1):
        self.key = key
        self.std = std

    def __call__(self, sample):
        sample[self.key] = Normal(sample[self.key], self.std).sample()
        return sample


class BinomialSubsample:
    def __init__(self, key="gene_counts", p=0.75, clamp_1=True):
        self.clamp = clamp_1
        self.key = key
        self.p = torch.tensor(p)

    def __call__(self, sample):
        resampled = Binomial(sample[self.key], self.p).sample()
        if self.clamp:
            resampled = torch.clamp(resampled, min=1)
        sample[self.key] = resampled
        return sample


class Mask:
    def __init__(self, p=0.15, key="gene_counts"):
        self.p = p
        self.key = key

    def __call__(self, sample):
        replace = sample[self.key].float()
        to_mask = torch.rand_like(replace) < self.p
        replace[to_mask] = torch.nan
        sample[self.key] = replace
        return sample


class MolecularCV:
    def __init__(self, p_to_train=0.8):
        self.p = torch.tensor(p_to_train)

    def __call__(self, sample):
        resampled = Binomial(sample["gene_counts"], self.p).sample()
        sample["gene_counts_true"] = sample["gene_counts"] - resampled
        sample["gene_counts"] = resampled
        return sample


class SequentialPreprocessor:
    def __init__(self, *args):
        self.preprocessors = args

    def __call__(self, sample):
        for step in self.preprocessors:
            sample = step(sample)
        return sample


def batch_collater(batch):
    batch_collated = {}

    batch_collated["0/obs"] = torch.tensor(np.array([b["0/obs"] for b in batch]))
    batch_collated["0/split"] = [b["0/split"] for b in batch]

    for name, padval in zip(
        ["gene_index", "gene_counts", "gene_counts_true"], [0, -1, -1]
    ):

        if batch[0][name].ndim == 1:
            if len({b[name].shape for b in batch}) == 1:
                batch_collated[k] = torch.stack([b[name] for b in batch])
            else:
                batch_collated[name] = pad_sequence(
                    [b[name] for b in batch], batch_first=True, padding_value=padval
                )
        else:
            if len({b[name].shape for b in batch}) == 1:
                batch_collated[k] = torch.cat([b[name] for b in batch], 0)
            batch_collated[name] = rearrange(
                pad_sequence(
                    [b[name].T for b in batch], batch_first=True, padding_value=padval
                ),
                "b l k -> (b k) l",
            )
    return batch_collated
