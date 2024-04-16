import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from express.utils.config import Config
from express import loss
from scipy.stats import spearmanr
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

        self.perturbation_indicator = nn.Parameter(torch.empty(dim_per_gene).uniform_(-1, 1))

        self.to_embed = nn.Linear(1, dim_per_gene) # B, G, 1 -> B, G, H1
        self.mixer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(dim_per_gene),
            Permute(0,2,1), # B, G, H1 -> B, H1, G
            nn.Linear(5000, bottleneck_dim), # B, H1, G -> B, H1, H2
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(bottleneck_dim),
            nn.Linear(bottleneck_dim, 5000), # B, H1, H2 -> B, H1, 5000
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.LayerNorm(5000),
            Permute(0,2,1), # B, H1, 5000 -> B, 5000, H1
            nn.Linear(dim_per_gene, 1), # B, 5000, 1

        )

        self.lr = self.config.lr
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
        
        loss = F.mse_loss(y, batch["gene_counts_true"].to(y.dtype))

        self.log("train_loss", loss , sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        
        loss = F.mse_loss(y, batch["gene_counts_true"].to(y.dtype))

        self.log("val_loss", loss, sync_dist=True)

        self.validation_step_outputs.append((y.cpu(), batch["gene_counts_true"].cpu(), batch["gene_counts_copy"].cpu()))

    def on_validation_epoch_end(self):
        all_preds = torch.cat([s[0] for s in self.validation_step_outputs])
        all_trues = torch.cat([s[1] for s in self.validation_step_outputs])
        all_origs = torch.cat([s[2] for s in self.validation_step_outputs])
        true_expr_change = all_trues - all_origs
        pred_expr_change = all_preds - all_origs

        delta_spearmans = []
        for i in range(len(true_expr_change)):
            s = spearmanr(pred_expr_change[i].float().numpy(), true_expr_change[i].float().numpy()).statistic
            delta_spearmans.append(s)
        self.log("val_deltaspearman(IC)", np.mean(delta_spearmans))

        spearmans = []
        for i in range(len(true_expr_change)):
            s = spearmanr(all_preds[i].float().numpy(), all_trues[i].float().numpy()).statistic
            spearmans.append(s)
        self.log("val_spearman(IC)", np.mean(spearmans))

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        return (y, batch["gene_counts_true"], batch["gene_counts_copy"])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer


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
            dim_to_loss = 2000
        else:
            layers = [2000] + layers
            net = []
            for i in range(len(layers)-1):
                net.append(nn.Linear(layers[i], layers[i+1]))
                net.append(nn.Dropout(0.20))
                net.append(nn.LayerNorm(layers[i+1]))
                net.append(nn.ReLU())
            self.net = nn.Sequential(*net)
            dim_to_loss = layers[-1]

        
        if self.config.celltype_clf_loss:
            self.loss = loss.CellTypeClfLoss(dim_to_loss, self.config.cls_finetune_dim)
        elif self.config.modality_prediction_loss:
            self.loss = loss.ModalityPredictionLoss(dim_to_loss, self.config.cls_finetune_dim)
        else:
            raise ValueError("At least one of celltype clf loss or modality predict loss should be true")

        self.lr = self.config.lr
        self.validation_step_outputs = []

        if self.config.celltype_clf_loss:
            self.micro_acc = MulticlassAccuracy(num_classes=self.config.cls_finetune_dim, average="micro")
            self.macro_acc = MulticlassAccuracy(num_classes=self.config.cls_finetune_dim, average="macro")

    def forward(self, batch):
        z = self.net(batch["gene_counts"])
        return self.loss.predict(z)


    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        
        loss = self.loss.loss(y, batch["0/targets"])

        self.log("train_loss", loss , sync_dist=True)
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
            spearmans = []
            for i in range(all_preds.shape[1]):
                s = spearmanr(all_preds[:, i].float().numpy(), all_trues[:, i].float().numpy()).statistic
                spearmans.append(s)
            self.log("val_macro_spearman", np.mean(spearmans), sync_dist=True)

        self.validation_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer