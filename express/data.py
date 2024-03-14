import h5torch
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from importlib.resources import files


class ExpressDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        bin_path,
        batch_size=16,  # batch size for model
        n_workers=4,  # num workers in dataloader
        in_memory=False,  # whether to use h5torch in-memory mode for more-efficient dataloading
        input_processing=[
            "countsper_x",
            "logp1",
        ],  # "countsper_x", "logp1", "norm_x", "bin"
    ):
        super().__init__()
        self.n_workers = n_workers
        self.bin_path = bin_path
        self.batch_size = batch_size
        self.path = path
        self.in_memory = in_memory
        self.input_processing = input_processing

    def setup(self, stage):

        mapper = {
            "countsper": lambda x: CountsPerX(factor=x, key="gene_counts"),
            "norm": lambda x: FixedNorm(factor=x, key="gene_counts"),
            "bin": lambda: Bin(bins=np.loadtxt(self.bin_path), key="gene_counts"),
            "logp1": lambda: LogP1(key="gene_counts"),
        }
        processor = SequentialPreprocessor(
            *[
                (
                    mapper[i.split("_")[0]](float(i.split("_")[1]))
                    if "_" in i
                    else mapper[i]()
                )
                for i in self.input_processing
            ],
            Mask(p=0.15, key="gene_counts"),
            Bin(bins=np.loadtxt(self.bin_path), key="gene_counts_true"),
            FilterMaxGenes(
                number=1024,
                affected_keys=["gene_counts", "gene_index", "gene_counts_true"],
            ),
        )

        self.train = h5torch.Dataset(
            self.path,
            sample_processor=CellSampleProcessor(processor),
            in_memory=self.in_memory,
            subset=("0/split", "train"),
        )

        self.val = h5torch.Dataset(
            self.path,
            sample_processor=CellSampleProcessor(processor),
            in_memory=self.in_memory,
            subset=("0/split", "val"),
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val,
            num_workers=self.n_workers,
            batch_size=self.batch_size,
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
                "gene_counts": gene_counts,
                "gene_index": gene_index,
                "gene_counts_true": gene_counts,
            }

        _ = sample.pop("central")
        sample = self.processor(sample)
        return sample

class RankCounts:
    def __init__(self, key = "gene_counts"):
        self.key = key

    def __call__(self, sample):
        sample[self.key] = np.argsort(sample[self.key])
        return sample


class CountsPerX:
    def __init__(self, factor=10_000, key="gene_counts"):
        self.factor = factor
        self.key = key

    def __call__(self, sample):
        sample[self.key] = sample[self.key] / sample[self.key].sum() * self.factor
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
        sample[self.key] = np.log1p(sample[self.key])
        return sample


class Bin:
    def __init__(self, bins, key="gene_counts"):
        self.bins = bins
        self.key = key

    def __call__(self, sample):
        sample[self.key] = np.digitize(sample[self.key], self.bins) - 1
        return sample

    @staticmethod
    def generate_bins(x, n):
        bins = np.quantile(x, np.linspace(0, 1, n + 1))
        bins = np.unique(bins)
        bins[-1] = bins[-1] + 1
        return bins


class FilterMaxGenes:
    def __init__(self, number=2048, affected_keys=["gene_counts", "gene_index"]):
        self.n = number
        self.affected_keys = affected_keys

    def __call__(self, sample):
        for a in self.affected_keys:
            sample[a] = sample[a][: self.n]
        return sample


class Mask:
    def __init__(self, p, key="gene_counts"):
        self.p = p
        self.key = key

    def __call__(self, sample):
        replace = sample[self.key].astype(float)
        to_mask = np.random.rand(len(replace)) < self.p
        replace[to_mask] = torch.nan
        sample[self.key] = replace
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
    keys = list(batch[0])
    for k in keys:
        v = [b[k] for b in batch]
        if isinstance(v[0], str):
            batch_collated[k] = v
        elif isinstance(v[0], (int, np.int64)):
            batch_collated[k] = torch.tensor(v)
        elif isinstance(v[0], np.ndarray):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.tensor(np.array(v))
            else:
                if k == "gene_index":
                    batch_collated[k] = pad_sequence(
                        [torch.tensor(t) for t in v], batch_first=True, padding_value=0
                    )
                else:
                    batch_collated[k] = pad_sequence(
                        [torch.tensor(t) for t in v], batch_first=True, padding_value=-1
                    )
        elif torch.is_tensor(v[0]):
            if len({t.shape for t in v}) == 1:
                batch_collated[k] = torch.stack(v)
            else:
                if v[0].dtype == torch.bool:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=False
                    )
                else:
                    batch_collated[k] = pad_sequence(
                        v, batch_first=True, padding_value=-1
                    )
    return batch_collated
