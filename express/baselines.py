import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from express.utils.config import Config
from express import loss
from scipy.stats import spearmanr
import numpy as np

class PerturbMixer(pl.LightningModule):
    def __init__(
        self,
        config_path,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = Config(config_path)
        dim_per_gene = self.config.baseline_perturb_dim_per_gene
        bottleneck_dim = self.config.baseline_perturb_bottleneck_dim

        self.perturbation_indicator = nn.Parameter(torch.empty(dim_per_gene).uniform_(-1, 1))

        self.to_embed = nn.Linear(1, dim_per_gene)
        self.mixer = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.BatchNorm1d(dim_per_gene*5000),
            nn.Linear(dim_per_gene*5000, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.BatchNorm1d(bottleneck_dim),
            nn.Linear(bottleneck_dim, 5000),
        )

        self.lr = self.config.lr
        self.validation_step_outputs = []

    def forward(self, batch):
        x = self.to_embed(batch["gene_counts"].unsqueeze(-1))

        matches = torch.where((batch["gene_index"].T == batch["0/perturbed_gene"]).T)
        assert (matches[0] == torch.arange(len(x)).to(matches[0].device)).all()
        x[torch.arange(len(x)), matches[1]] += self.perturbation_indicator

        x = x.view(len(x), -1)
        z = self.mixer(x)

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
        self.log("val_deltaspearman", np.mean(delta_spearmans))

        spearmans = []
        for i in range(len(true_expr_change)):
            s = spearmanr(all_preds[i].float().numpy(), all_trues[i].float().numpy()).statistic
            spearmans.append(s)
        self.log("val_spearman", np.mean(spearmans))

        self.validation_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        return (y, batch["gene_counts_true"], batch["gene_counts_copy"])


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer