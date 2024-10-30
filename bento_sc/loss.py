import torch
import torch.nn as nn
import torch.nn.functional as F


class BentoLoss(nn.Module):
    def __init__(self, in_dim, out_dim, reduction="mean"):
        super().__init__()

        self.output_head = nn.Linear(in_dim, out_dim)

        if reduction == "mean":
            self.reduce = torch.mean
        elif reduction == "sum":
            self.reduce = torch.sum
        elif reduction == "none":
            self.reduce = lambda x: x

    def predict(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        raise NotImplementedError

    def loss(self, *args): 
        raise NotImplementedError


class BinCE(BentoLoss):
    def __init__(self, dim, n_bins, reduction="mean"):
        super().__init__(dim, n_bins, reduction=reduction)

    def predict(self, x, **kwargs):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x, targets, train_on=None, **kwargs):
        if train_on is not None:
            y = self.predict(x)[train_on]
            return self.loss(y, targets[train_on])
        else:
            y = self.predict(x)
            return self.loss(y, targets)


class CountMSE(BentoLoss):
    def __init__(self, dim, exp_output=True, lib_norm=False, reduction="mean", plus_one=False):
        super().__init__(dim, 1, reduction=reduction)
        assert not (not exp_output and lib_norm), "lib norm true needs exp output True"
        self.exp_output = exp_output
        self.lib_norm = lib_norm
        self.plus_one = plus_one

    def predict(self, x, libsize=None, **kwargs):
        y = self.output_head(x).squeeze(-1)
        if not self.exp_output:
            return y
        if not self.plus_one:
            if self.lib_norm:
                return F.softmax(y, -1) * libsize
            else:
                return torch.clamp(y.exp(), max=1e7)
        else:
            if self.lib_norm:
                return F.softmax(y, -1) * (libsize - y.shape[-1]) + 1
            else:
                return torch.clamp(y.exp() + 1, max=1e7)

    def loss(self, inputs, targets):
        return self.reduce(F.mse_loss(inputs, targets, reduction="none"))

    def forward(self, x, targets, train_on=None, **kwargs):
        y = self.predict(x, libsize=targets.sum(1)[:, None])
        if train_on is not None:
            return self.loss(y[train_on], targets.to(y.dtype)[train_on])
        else:
            return self.loss(y, targets.to(y.dtype))


class PoissonNLL(BentoLoss):
    def __init__(
        self,
        dim,
        lib_norm=False,
        reduction="mean",
        omit_last_term=True,
        zero_truncated=False,
    ):
        super().__init__(dim, 1, reduction=reduction)
        self.omit = omit_last_term
        self.lib_norm = lib_norm
        self.zero_trunc = zero_truncated

    def predict(self, x, libsize=None,  **kwargs):
        y = self.output_head(x).squeeze(-1)
        if self.zero_trunc:
            if self.lib_norm:
                return F.softmax(y, -1) * (libsize - y.shape[-1]) + 1
            else:
                return torch.clamp(y.exp() + 1, max=1e7)
        else:
            if self.lib_norm:
                return F.softmax(y, -1) * libsize
            else:
                return torch.clamp(y.exp(), max=1e7)

    def loss(self, inputs, targets):
        if self.zero_trunc:
            stabilized_term = torch.where(
                inputs < 10, (torch.clamp(inputs,max=10).exp() - 1 + 1e-8).log(), inputs
            )
            loss = stabilized_term - targets * (inputs + 1e-8).log()
        else:
            loss = inputs - targets * (inputs + 1e-8).log()

        if self.omit:
            return self.reduce(loss)
        else:
            return self.reduce(loss + torch.lgamma(targets + 1))

    def forward(self, x, targets, train_on=None, **kwargs):
        y = self.predict(x, libsize=targets.sum(1)[:, None])
        if train_on is not None:
            return self.loss(y[train_on], targets[train_on])
        else:
            return self.loss(y, targets)


class NegativeBinomialNLL(BentoLoss):
    def __init__(
        self,
        dim,
        lib_norm=False,
        n_genes=19331,
        fixed_dispersion=False,
        reduction="mean",
        omit_last_term=True,
        zero_truncated=False,
    ):
        super().__init__(dim, (1 if fixed_dispersion else 2), reduction=reduction)

        self.omit = omit_last_term
        self.fixed_dispersion = fixed_dispersion
        if self.fixed_dispersion:
            self.dispersions = nn.Embedding(n_genes, 1)
        self.lib_norm = lib_norm
        self.zero_trunc = zero_truncated

    def predict(self, x, gene_ids=None, libsize=None, **kwargs):
        y = self.output_head(x)
        if self.fixed_dispersion:
            mus = y.squeeze(-1)
            log_thetas = self.dispersions(gene_ids).squeeze(-1)
        else:
            mus, log_thetas = y[..., 0], y[..., 1]

        if self.zero_trunc:
            if self.lib_norm:
                mus = F.softmax(mus, -1) * (libsize - mus.shape[-1]) + 1
            else:
                mus = torch.clamp(mus.exp() + 1, max=1e7)
        else:
            if self.lib_norm:
                mus = F.softmax(mus, -1) * libsize
            else:
                mus = torch.clamp(mus.exp(), max=1e7)

        return mus, log_thetas

    def loss(self, mus, log_thetas, targets):

        eps = torch.finfo(mus.dtype).tiny

        log_thetas += eps
        thetas = log_thetas.exp()
        mus += eps
        
        loss = (
            torch.lgamma(thetas.float())
            - torch.lgamma((targets + thetas).float())
            + targets * (thetas + mus).log()
            - thetas * log_thetas
            - targets * mus.log()
        ).to(mus.dtype)
        if self.zero_trunc:
            stabilized_term = torch.where(
                torch.logical_or(
                    (mus-1+1e-8)*(thetas+1e-8) > 15,
                    mus+thetas>15
                ),
                thetas * (thetas + mus).log(),
                (torch.clamp(thetas + mus, max=15) ** torch.clamp(thetas, max=15) - 
                 torch.clamp(thetas, max=15)**torch.clamp(thetas, max=15) + 1e-8).log(),
            )
            loss += stabilized_term
        else:
            loss += thetas * (thetas + mus).log()

        if self.omit:
            return self.reduce(loss)
        else:
            return self.reduce(loss + torch.lgamma(targets + 1))

    def forward(self, x, targets, gene_ids=None, train_on=None):
        mus, log_thetas = self.predict(
            x, gene_ids=gene_ids, libsize=targets.sum(1)[:, None]
        )

        if train_on is not None:
            return self.loss(mus[train_on], log_thetas[train_on], targets[train_on])
        else:
            return self.loss(mus, log_thetas, targets)


class ZeroInflatedNegativeBinomialNLL(NegativeBinomialNLL):
    def __init__(
        self,
        dim,
        lib_norm=False,
        n_genes=19331,
        fixed_dispersion=False,
        reduction="mean",
        omit_last_term=True,
    ):
        super().__init__(
            dim,
            lib_norm=lib_norm,
            n_genes=n_genes,
            fixed_dispersion=fixed_dispersion,
            reduction=reduction,
            omit_last_term=omit_last_term,
        )

        self.out_pi = nn.Linear(dim, 1)

    def predict(self, x, gene_ids=None, libsize=None, **kwargs):
        mus, log_thetas = super().predict(x, gene_ids=gene_ids, libsize=libsize)
        pis = self.out_pi(x).squeeze(-1)
        return mus, log_thetas, pis

    def loss(self, mus, log_thetas, pis, targets):
        NB_NLL_loss = super().loss(mus, log_thetas, targets)

        eps = torch.finfo(mus.dtype).tiny
        mus += eps
        log_thetas += eps
        thetas = log_thetas.exp()

        indices = targets > 0

        NLLifzero = F.softplus(-pis) - F.softplus(
            -pis + thetas * (log_thetas - (thetas + mus).log())
        )
        NLLifnotzero = pis + F.softplus(-pis) + NB_NLL_loss

        return self.reduce(NLLifnotzero * indices + NLLifzero * ~indices)

    def forward(self, x, targets, gene_ids=None):
        mus, log_thetas, pis = self.predict(x, gene_ids=gene_ids, libsize=targets.sum())
        return self.loss(mus, log_thetas, pis, targets)


class NCELoss(BentoLoss):
    def __init__(
        self,
        dim,
        embed_dim,
        reduction="mean",
        temperature=1,
    ):
        super().__init__(dim, embed_dim, reduction=reduction)
        self.t = temperature

    def predict(self, x, **kwargs):
        return self.output_head(x)

    def loss(self, inputs):
        n = len(inputs)
        targets = torch.arange(n).view(n // 2, 2).fliplr().view(n)

        inputs = F.normalize(inputs, dim = -1)
        inputs = (inputs @ inputs.T) / self.t
        inputs.fill_diagonal_(float("-inf"))
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x):
        y = self.predict(x)
        return self.loss(y)


class CellTypeClfLoss(BentoLoss):
    def __init__(
        self,
        dim,
        n_classes,
        reduction="mean",
    ):
        super().__init__(dim, n_classes, reduction=reduction)

    def predict(self, x, **kwargs):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x, targets):
        y = self.predict(x)
        return self.loss(y, targets)

class ModalityPredictionLoss(BentoLoss):
    def __init__(
        self,
        dim,
        n_classes,
        reduction="mean",
    ):
        super().__init__(dim, n_classes, reduction=reduction)

    def predict(self, x, **kwargs):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.mse_loss(inputs, torch.log1p(targets.to(inputs.dtype)), reduction="none"))

    def forward(self, x, targets):
        y = self.predict(x)
        return self.loss(y, targets)