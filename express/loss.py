import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpressLoss(nn.Module):
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


class BinCE(ExpressLoss):
    def __init__(self, dim, n_bins, reduction="mean"):
        super().__init__(dim, n_bins, reduction=reduction)

    def predict(self, x):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x, targets, **kwargs):
        y = self.predict(x)
        return self.loss(y, targets)


class CountMSE(ExpressLoss):
    def __init__(self, dim, exp_output=True, lib_norm=False, reduction="mean"):
        super().__init__(dim, 1, reduction=reduction)
        assert not (not exp_output and lib_norm), "lib norm true needs exp output True"
        self.exp_output = exp_output
        self.lib_norm = lib_norm

    def predict(self, x, libsize=None):
        y = self.output_head(x).squeeze(-1)
        if not self.exp_output:
            return y
        elif self.lib_norm:
            return F.softmax(y, -1) * libsize
        else:
            return y.exp()

    def loss(self, inputs, targets):
        return self.reduce(F.mse_loss(inputs, targets, reduction="none"))

    def forward(self, x, targets, **kwargs):
        y = self.predict(x, libsize=targets.sum())
        return self.loss(y, targets.to(y.dtype))


class PoissonNLL(ExpressLoss):
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

    def predict(self, x, libsize=None):
        y = self.output_head(x).squeeze(-1)
        if self.lib_norm:
            return F.softmax(y, -1) * libsize
        else:
            return y.exp()

    def loss(self, inputs, targets):
        if self.zero_trunc:
            stabilized_term = torch.where(
                inputs < 10, (inputs.exp() - 1 + 1e-8).log(), inputs
            )
            loss = stabilized_term - targets * (inputs + 1e-8).log()
        else:
            loss = inputs - targets * (inputs + 1e-8).log()

        if self.omit:
            return self.reduce(loss)
        else:
            return self.reduce(loss + torch.lgamma(targets + 1))

    def forward(self, x, targets, **kwargs):
        y = self.predict(x, libsize=targets.sum())
        return self.loss(y, targets)


class NegativeBinomialNLL(ExpressLoss):
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

    def predict(self, x, gene_ids=None, libsize=None):
        y = self.output_head(x)
        if self.fixed_dispersion:
            mus = y.squeeze(-1)
            log_thetas = self.dispersions(gene_ids).squeeze(-1)
        else:
            mus, log_thetas = y[..., 0], y[..., 1]

        if self.lib_norm:
            mus = F.softmax(mus, -1) * libsize
        else:
            mus = mus.exp()

        return mus, log_thetas

    def loss(self, mus, log_thetas, targets):

        eps = torch.finfo(mus.dtype).tiny

        log_thetas += eps
        thetas = log_thetas.exp()
        mus += eps

        loss = (
            torch.lgamma(thetas)
            - torch.lgamma(targets + thetas)
            + targets * (thetas + mus).log()
            - thetas * log_thetas
            - targets * mus.log()
        )
        if self.zero_trunc:
            stabilized_term = torch.where(
                torch.logical_and(thetas > 10, mus > 10),
                thetas * (thetas + mus).log(),
                ((thetas + mus) ** thetas - thetas**thetas).log(),
            )
            loss += stabilized_term
        else:
            loss += thetas * (thetas + mus).log()

        if self.omit:
            return self.reduce(loss)
        else:
            return self.reduce(loss + torch.lgamma(targets + 1))

    def forward(self, x, targets, gene_ids=None):
        mus, log_thetas = self.predict(
            self, x, gene_ids=gene_ids, libsize=targets.sum()
        )
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

    def predict(self, x, gene_ids=None, libsize=None):
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


class NCELoss(ExpressLoss):
    def __init__(
        self,
        dim,
        embed_dim,
        reduction="mean",
        temperature=1,
    ):
        super().__init__(dim, embed_dim, reduction=reduction)
        self.t = temperature

    def predict(self, x):
        return self.output_head(x)

    def loss(self, inputs):
        n = len(inputs)
        targets = torch.arange(n).view(n // 2, 2).fliplr().view(n)

        inputs = F.normalize(inputs)
        inputs = (inputs @ inputs.T) / self.t
        inputs.fill_diagonal_(float("-inf"))
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x):
        y = self.predict(x)
        return self.loss(y)


class CellTypeClfLoss(ExpressLoss):
    def __init__(
        self,
        dim,
        n_classes,
        reduction="mean",
    ):
        super().__init__(dim, n_classes, reduction=reduction)

    def predict(self, x):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.cross_entropy(inputs, targets, reduction="none"))

    def forward(self, x, targets):
        y = self.predict(x)
        return self.loss(y, targets)

class ModalityPredictionLoss(ExpressLoss):
    def __init__(
        self,
        dim,
        n_classes,
        reduction="mean",
    ):
        super().__init__(dim, n_classes, reduction=reduction)

    def predict(self, x):
        return self.output_head(x)

    def loss(self, inputs, targets):
        return self.reduce(F.mse_loss(inputs, targets.to(inputs.dtype), reduction="none"))

    def forward(self, x, targets):
        y = self.predict(x)
        return self.loss(y, targets)