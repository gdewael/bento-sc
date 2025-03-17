import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from bento_sc.utils.config import Config
from bento_sc.utils.metrics import pearson_batch_masked
from bento_sc import loss
from scipy.stats import pearsonr
import numpy as np
from torchmetrics.classification import MulticlassAccuracy
from copy import deepcopy


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.permute(*self.args)


class View(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.reshape(*self.args)


class PerturbBaseline(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = deepcopy(config)
        dim_per_gene = self.config.baseline_perturb_dim_per_gene
        bottleneck_dim = self.config.baseline_perturb_bottleneck_dim

        self.perturbation_indicator = nn.Parameter(
            torch.empty(dim_per_gene).uniform_(-1, 1) / self.config.perturb_init_factor
        )

        self.to_embed = nn.Linear(1, dim_per_gene)  # B, G, 1 -> B, G, H1
        self.mixer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(dim_per_gene),
            Permute(0, 2, 1),  # B, G, H1 -> B, H1, G
            nn.Linear(5000, bottleneck_dim),  # B, H1, G -> B, H1, H2
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, 5000),  # B, H1, H2 -> B, H1, 5000
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(5000),
            Permute(0, 2, 1),  # B, H1, 5000 -> B, 5000, H1
            nn.Linear(dim_per_gene, 1),  # B, 5000, 1
        )

        self.lr = float(self.config.lr)
        self.validation_step_outputs = []

    def forward(self, batch):
        x = self.to_embed(batch["gene_counts"].unsqueeze(-1))

        matches = torch.where((batch["gene_index"].T == batch["0/perturbed_gene"]).T)
        assert (matches[0] == torch.arange(len(x)).to(matches[0].device)).all()
        x[torch.arange(len(x)), matches[1]] += self.perturbation_indicator

        z = self.mixer(x).squeeze(-1)
        return z

    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)

        target = batch["gene_counts_true"]

        loss = F.mse_loss(y, target.to(y.dtype))

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)

        target = batch["gene_counts_true"]

        loss = F.mse_loss(y, target.to(y.dtype))

        self.log("val_loss", loss, sync_dist=True)

        self.validation_step_outputs.append(
            (
                y.cpu(),
                batch["gene_counts_true"].cpu(),
                batch["gene_counts_copy"].cpu(),
                batch["0/perturbed_gene"].cpu(),
            )
        )

    def on_validation_epoch_end(self):
        pred_pert_prof = torch.cat([s[0] for s in self.validation_step_outputs])
        true_pert_prof = torch.cat([s[1] for s in self.validation_step_outputs])
        perturbed_genes = torch.cat([s[3] for s in self.validation_step_outputs])

        mean_ctrl_profile = torch.cat(
            [s[2] for s in self.validation_step_outputs]
        ).mean(0)

        pearsons = []
        delta_pearsons = []
        for pert_gene in torch.unique(perturbed_genes):
            pred_per_pert = pred_pert_prof[perturbed_genes == pert_gene].float().mean(0)
            true_per_pert = true_pert_prof[perturbed_genes == pert_gene].float().mean(0)

            pred_delta_per_pert = pred_per_pert - mean_ctrl_profile
            true_delta_per_pert = true_per_pert - mean_ctrl_profile

            s1 = pearsonr(pred_per_pert.numpy(), true_per_pert.numpy()).statistic
            pearsons.append(s1)

            s2 = pearsonr(
                pred_delta_per_pert.numpy(), true_delta_per_pert.numpy()
            ).statistic
            delta_pearsons.append(s2)
        self.log("val_pearson", np.mean(pearsons), sync_dist=True)
        self.log("val_deltapearson", np.mean(delta_pearsons), sync_dist=True)

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        return (
            y,
            batch["gene_counts_true"],
            batch["gene_counts_copy"],
            batch["gene_index"],
        )

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    @property
    def config_used(self):
        return {
            "baseline_perturb_dim_per_gene",
            "baseline_perturb_bottleneck_dim",
            "lr",
        }

    @property
    def config_unused(self):
        return set(self.config) - self.config_used


class CLSTaskBaseline(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = deepcopy(config)
        layers = self.config.baseline_cls_task_layers

        if layers == []:
            self.net = nn.Identity()
            dim_to_loss = self.config.baseline_cls_hvg
        else:
            layers = [self.config.baseline_cls_hvg] + layers
            net = []
            for i in range(len(layers) - 1):
                net.append(nn.Linear(layers[i], layers[i + 1]))
                net.append(nn.Dropout(0.20))
                net.append(nn.LayerNorm(layers[i + 1]))
                net.append(nn.ReLU())
            self.net = nn.Sequential(*net)
            dim_to_loss = layers[-1]

        if self.config.celltype_clf_loss:
            self.loss = loss.CellTypeClfLoss(dim_to_loss, self.config.cls_finetune_dim)
        elif self.config.modality_prediction_loss:
            self.loss = loss.ModalityPredictionLoss(
                dim_to_loss, self.config.cls_finetune_dim
            )
        else:
            raise ValueError(
                "At least one of celltype clf loss or modality predict loss should be true"
            )

        self.lr = float(self.config.lr)
        self.validation_step_outputs = []

        if self.config.celltype_clf_loss:
            self.micro_acc = MulticlassAccuracy(
                num_classes=self.config.cls_finetune_dim, average="micro"
            )
            self.macro_acc = MulticlassAccuracy(
                num_classes=self.config.cls_finetune_dim, average="macro"
            )

    def forward(self, batch):
        z = self.net(batch["gene_counts"])
        return self.loss.predict(z)

    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)

        loss = self.loss.loss(y, batch["0/targets"])

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)

        loss = self.loss.loss(y, batch["0/targets"])

        self.log("val_loss", loss, sync_dist=True)
        if self.config.celltype_clf_loss:
            self.micro_acc(y, batch["0/targets"])
            self.macro_acc(y, batch["0/targets"])
            self.log(
                "val_microacc",
                self.micro_acc,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch["0/targets"]),
                sync_dist=True,
            )
            self.log(
                "val_macroacc",
                self.macro_acc,
                on_step=False,
                on_epoch=True,
                batch_size=len(batch["0/targets"]),
                sync_dist=True,
            )

        self.validation_step_outputs.append((y.cpu(), batch["0/targets"].cpu()))

    def on_validation_epoch_end(self):
        all_preds = torch.cat([s[0] for s in self.validation_step_outputs])
        all_trues = torch.cat([s[1] for s in self.validation_step_outputs])

        if isinstance(self.loss, loss.ModalityPredictionLoss):
            pearson_per_target = pearson_batch_masked(
                all_preds.T.float(), all_trues.T.float()
            ).numpy()
            self.log("val_macro_pearson", np.mean(pearson_per_target), sync_dist=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer

    @property
    def config_used(self):
        return {
            "baseline_cls_task_layers",
            "celltype_clf_loss",
            "modality_prediction_loss",
            "cls_finetune_dim",
            "lr",
        }

    @property
    def config_unused(self):
        return set(self.config) - self.config_used


class VAE(pl.LightningModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = deepcopy(config)
        layers = self.config.vae_layers

        net = []
        for i in range(len(layers) - 1):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Dropout(0.20))
            net.append(nn.LayerNorm(layers[i + 1]))
            net.append(nn.ReLU())
        self.encoder = nn.Sequential(*net)

        net = []
        for i in range(len(layers) - 2, 0, -1):
            net.append(nn.Linear(layers[i + 1], layers[i]))
            net.append(nn.Dropout(0.20))
            net.append(nn.LayerNorm(layers[i]))
            net.append(nn.ReLU())
        net.append(nn.Linear(layers[1], layers[0]))

        self.decoder = nn.Sequential(*net)

        self.to_mu = nn.Linear(layers[-1], layers[-1])
        self.to_var = nn.Linear(layers[-1], layers[-1])
        self.beta = self.config.beta

        self.lr = float(self.config.lr)

    def forward(self, batch):
        z = self.encoder(batch["gene_counts"])
        z_means, z_logvars = self.to_mu(z), self.to_var(z)

        z = self.reparameterization(z_means, z_logvars)
        x_reconstruct = self.decoder(z)

        return x_reconstruct, z_means, z_logvars

    def reparameterization(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def KL_div(self, means, logvars):
        KL_div_loss = (
            -0.5 * torch.sum(1 + logvars - means**2 - logvars.exp())
        ) / means.size(1)
        return KL_div_loss

    def MSE(self, inputs, targets):
        return F.mse_loss(inputs, targets, reduction="none")

    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y, means, logvars = self(batch)
        pred_counts = y

        recon_loss = self.MSE(pred_counts, batch["gene_counts_true"])
        kl_div = self.KL_div(means, logvars)
        loss = (recon_loss + self.beta * kl_div).mean()

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y, means, logvars = self(batch)
        pred_counts = y

        recon_loss = self.MSE(pred_counts, batch["gene_counts_true"])
        kl_div = self.KL_div(means, logvars)
        loss = (recon_loss + self.beta * kl_div).mean()

        self.log("val_loss", loss, sync_dist=True)
        return loss

    def predict_step(self, batch):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        z = self.encoder(batch["gene_counts"])
        z_means = self.to_mu(z)

        y = self.decoder(z_means)

        count_predictions = torch.clamp(y.exp(), max=1e7)

        return (batch["0/obs"], z_means, count_predictions, batch["gene_counts_true"])

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
