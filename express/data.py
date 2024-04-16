import h5torch
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from lightning import LightningDataModule
from importlib.resources import files
from torch.distributions.binomial import Binomial
from torch.distributions.poisson import Poisson
from torch.distributions.normal import Normal
from einops import rearrange
from h5torch.dataset import sample_csr

class ExpressDataModule(LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

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
            "CountsAsPositions": CountsAsPositions,
            "Copy": Copy,
            "FilterHVG": FilterHVG,
        }

        if "input_processing" in self.config:
            processor = []
            for prepr in self.config["input_processing"]:
                type_ = prepr.pop("type")
                processor.append(preprocessing_mapper[type_](**prepr))
        else:
            processor = []
        processor = SequentialPreprocessor(*processor)

        if not self.config.perturb_mode:
            processing_class = CellSampleProcessor
        else:
            processing_class = PerturbationCellSampleProcessor

        f = h5torch.File(self.config["data_path"])

        if self.config["in_memory"]:
            f = f.to_dict()

        self.train = h5torch.Dataset(
            f,
            sample_processor=processing_class(
                processor, return_zeros=self.config["return_zeros"]
            ),
            subset=("0/split", "train"),
        )

        self.val = h5torch.Dataset(
            f,
            sample_processor=processing_class(
                processor, return_zeros=self.config["return_zeros"]
            ),
            subset=("0/split", "val"),
        )

        self.val_sub = h5torch.Dataset(
            f,
            sample_processor=processing_class(
                processor, return_zeros=self.config["return_zeros"]
            ),
            subset=("0/split", "val_sub"),
        )

        self.test = h5torch.Dataset(
            f,
            sample_processor=processing_class(
                processor, return_zeros=self.config["return_zeros"]
            ),
            subset=("0/split", "test"),
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
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )
    
    def val_sub_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_sub,
            num_workers=self.config["n_workers"],
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
            collate_fn=batch_collater,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test,
            num_workers=self.config["n_workers"],
            batch_size=self.config["batch_size"],
            shuffle=False,
            pin_memory=True,
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
            sample |= {
                "gene_counts": torch.tensor(gene_counts),
                "gene_counts_true": torch.tensor(gene_counts),
                "gene_index" : torch.arange(self.n_genes),
            }
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
        if self.processor is not None:
            sample = self.processor(sample)
        return sample
    

class PerturbationCellSampleProcessor:
    def __init__(self, processor, return_zeros=True, n_genes=19331):
        assert return_zeros == True, "return zeros has to be true for perturbation modeling."

        self.processor = processor
        self.processor_trues = SequentialPreprocessor(
            CountsPerX(factor=10_000, key="gene_counts_true"),
            LogP1(key="gene_counts_true"),

        )

        self.processor_origs = SequentialPreprocessor(
            Copy(key="gene_counts", to="gene_counts_copy"),
            CountsPerX(factor=10_000, key="gene_counts_copy"),
            LogP1(key="gene_counts_copy"),
        )

        self.n_genes = n_genes
        path = files("express.utils.data").joinpath("gene_set_perturb.txt")
        self.gene_indices = torch.tensor(np.loadtxt(path).astype(int))

    def __call__(self, f, sample):
        gene_counts = np.zeros(self.n_genes)
        gene_counts[sample["central"][0]] = sample["central"][1]
        sample |= {
                "gene_counts_true": torch.tensor(gene_counts),
                "gene_index" : torch.arange(self.n_genes),
            }
        
        if sample["0/split"] != "train":
            control_sample = sample_csr(f["central"], sample["0/matched_control"])

        else:
            train_control_indices = f["unstructured/train_control_indices"][:]
            sampled_control = np.random.choice(train_control_indices)
            control_sample = sample_csr(f["central"], sampled_control)

        gene_counts = np.zeros(self.n_genes)
        gene_counts[control_sample[0]] = control_sample[1]
        sample["gene_counts"] = torch.tensor(gene_counts)

        _ = sample.pop("central")

        sample = self.processor_origs(sample)

        if self.processor is not None:
            sample = self.processor(sample)

        sample = self.processor_trues(sample)

        sample["gene_counts_copy"] = sample["gene_counts_copy"][self.gene_indices]
        sample["gene_counts"] = sample["gene_counts"][self.gene_indices]
        sample["gene_counts_true"] = sample["gene_counts_true"][self.gene_indices]
        sample["gene_index"] = self.gene_indices

        return sample


class RankCounts:
    def __init__(self, key="gene_counts"):
        self.key = key

    def __call__(self, sample):
        sample[self.key] = torch.argsort(sample[self.key], descending=True)
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
        key_to_determine_top="gene_counts"
    ):
        self.n = number
        self.affected_keys = affected_keys
        self.topkey = key_to_determine_top

    def __call__(self, sample):
        if sample[self.affected_keys[0]].ndim == 1:
            to_select = torch.argsort(sample[self.topkey]).flip(0)[:self.n]
            for a in self.affected_keys:
                sample[a] = sample[a][to_select]
        else:
            to_select = torch.argsort(sample[self.topkey]).fliplr()[:, :self.n]
            for a in self.affected_keys:
                sample[a] = sample[a][torch.arange(2).unsqueeze(-1), to_select]
        return sample


class FilterHVG:
    def __init__(self, affected_keys=["gene_counts", "gene_index", "gene_counts_true"], number = 1024, dataset="cellxgene"):
        assert dataset in ["cellxgene", "citeseq", "greatapes"]
        path = files("express.utils.data").joinpath("hvg_%s.npy" % dataset)
        var = np.load(path)
        asort = np.argsort(var)[::-1][:number]
        self.to_select = torch.sort(torch.tensor(asort.copy())).values 
        self.affected_keys = affected_keys

    def __call__(self, sample):
        if sample[self.affected_keys[0]].ndim == 1:
            len_ = len(sample[self.affected_keys[0]])
            assert len_ == 19331
            indices = torch.argsort(sample["gene_index"])[self.to_select]
            for a in self.affected_keys:
                sample[a] = sample[a][indices]
        else:
            len_ = len(sample[self.affected_keys[0]][0])
            assert len_ == 19331
            indices = torch.argsort(sample["gene_index"])[:, self.to_select]
            for a in self.affected_keys:
                sample[a] = sample[a][torch.arange(2).unsqueeze(-1), indices]
        return sample


class FilterRandomGenes:
    def __init__(
        self,
        number=1024,
        affected_keys=["gene_counts", "gene_index", "gene_counts_true"],
        proportional_hvg = False,
    ):
        self.n = number
        self.affected_keys = affected_keys
        if proportional_hvg:
            path = files("express.utils.data").joinpath("hvg_cellxgene.npy")
            var = np.load(path)
            self.p = np.exp(np.log10(var+1e-8)/5)/np.exp(np.log10(var+1e-8)/5).sum()
        self.proportional_hvg = proportional_hvg

    def __call__(self, sample):
        if sample[self.affected_keys[0]].ndim == 1:
            len_ = len(sample[self.affected_keys[0]])
            indices = self._sample(sample, len_)
            for a in self.affected_keys:
                sample[a] = sample[a][indices]
        else:
            len_ = len(sample[self.affected_keys[0]][0])
            indices = torch.stack(
                [
                    self._sample(sample, len_),
                    self._sample(sample, len_),
                ]
            )
            for a in self.affected_keys:
                sample[a] = sample[a][torch.arange(2).unsqueeze(-1), indices]
        return sample
    
    def _sample(self, sample, len_):
        if not self.proportional_hvg:
            return torch.randperm(len_)[: self.n]
        else:
            assert len_ == 19331, "proportional hvg sampling can only be done on all genes"
            to_sample_gene_ix = torch.tensor(np.random.choice(len_, size=(self.n, ), p=self.p), replace=False)
            return torch.argsort(sample["gene_index"])[to_sample_gene_ix]


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
    
class Copy:
    def __init__(self, key="gene_counts", to="gene_counts_copy"):
        self.key = key
        self.to = to

    def __call__(self, sample):
        sample[self.to] = sample[self.key].clone()
        return sample
    
class CountsAsPositions:
    def __init__(self):
        """
        Used for Geneformer-style pre-training:
            - Gene index: RankCounts Gene counts
            - Gene counts true: Gene index
            - Gene counts: Gene index
        Make sure these have been run before performing this step:
            RankCounts gene counts
            FilterTop
        """
        pass
    def __call__(self, sample):
        gene_index = sample["gene_index"].clone()
        gene_counts = sample["gene_counts"].clone()
        sample["gene_index"] = gene_counts
        sample["gene_counts"] = gene_index
        sample["gene_counts_true"] = gene_index
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
    
    batch_collated["0/split"] = [b["0/split"] for b in batch]

    if "0/obs" in batch[0]:
        batch_collated["0/obs"] = torch.tensor(np.array([b["0/obs"] for b in batch]))

    if "0/perturbed_gene" in batch[0]:
        batch_collated["0/perturbed_gene"] = torch.tensor(np.array([b["0/perturbed_gene"] for b in batch])).long()

    if "0/ADT" in batch[0]:
        batch_collated["0/targets"] = torch.tensor(np.array([b["0/ADT"] for b in batch]))
    elif ("0/obs" in batch[0]) and (batch[0]["0/obs"].shape[0] == 9):
        # essentially hardcoded to detect the cellxgene pre-training set this way
        # (length 9 obs TODO make more elegant by changing the cellxgene processing script)
        batch_collated["0/targets"] = torch.tensor(np.array([b["0/obs"][3] for b in batch]))

    counts_keys = ["gene_index", "gene_counts", "gene_counts_true"]
    append_value = [0, -1, -1]
    if "gene_counts_copy" in batch[0]:
        counts_keys.append("gene_counts_copy")
        append_value.append(-1)

    for name, padval in zip(
        counts_keys, append_value
    ):

        if batch[0][name].ndim == 1:
            if len({b[name].shape for b in batch}) == 1:
                batch_collated[name] = torch.stack([b[name] for b in batch])
            else:
                batch_collated[name] = pad_sequence(
                    [b[name] for b in batch], batch_first=True, padding_value=padval
                )
        else:
            if len({b[name].shape for b in batch}) == 1:
                batch_collated[name] = torch.cat([b[name] for b in batch], 0)
            batch_collated[name] = rearrange(
                pad_sequence(
                    [b[name].T for b in batch], batch_first=True, padding_value=padval
                ),
                "b l k -> (b k) l",
            )

    if len(batch_collated["gene_index"]) != len(batch_collated["0/split"]):
        batch_collated["0/split"] = list(np.repeat(batch_collated["0/split"],2))
        if "0/obs" in batch_collated:
            batch_collated["0/obs"] = torch.repeat_interleave(batch_collated["0/obs"], 2, dim=0)
        
    return batch_collated
