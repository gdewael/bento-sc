import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import DiscreteEmbedding, ContinuousEmbedding
from express import loss
from scipy.stats import spearmanr
from torchmetrics.classification import MulticlassAccuracy
import numpy as np

class EmbeddingGater(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(dim).uniform_(-1, 1))

    def forward(self, x):
        y = F.tanh(x) * self.embedding
        y[:, 0, :] = x[:, 0, :]
        return y

class EmbeddingPseudoQuantizer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-1, 1))

    def forward(self, x):
        y = F.softmax(x, dim=-1) @ self.embedding
        y[:, 0, :] = x[:, 0, :]
        return y


class ExpressTransformer(pl.LightningModule):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.save_hyperparameters()

        self.config = config

        if self.config.discrete_input:
            self.embedder = DiscreteEmbedding(self.config.n_discrete_tokens, self.config.dim, cls=True)
        else:
            self.embedder = ContinuousEmbedding(self.config.dim, cls=True)

        if self.config.pseudoquant_input:
            self.embedder = nn.Sequential(self.embedder, EmbeddingPseudoQuantizer(self.config.dim, self.config.dim))
        elif self.config.gate_input:
            self.embedder = nn.Sequential(self.embedder, EmbeddingGater(self.config.dim))

        self.transformer = TransformerEncoder(
            depth=self.config.depth,
            dim=self.config.dim,
            nh=8,
            attentiontype="vanilla",
            attention_args={
                "dropout": self.config.dropout,
                "enable_math" : False,
                "enable_flash" : True,
                "enable_mem_efficient" : True
                },
            plugintype="learned",
            plugin_args={"dim": self.config.dim, "max_seq_len": self.config.n_genes},
            only_apply_plugin_at_first=True,
            dropout=self.config.dropout,
            glu_ff=True,
            activation="gelu",
        )
        
        loss_mapper = {
            "BinCE": loss.BinCE,
            "CountMSE": loss.CountMSE,
            "PoissonNLL": loss.PoissonNLL,
            "NegativeBinomialNLL": loss.NegativeBinomialNLL,
            "ZeroInflatedNegativeBinomialNLL": loss.ZeroInflatedNegativeBinomialNLL,
        }

        type_ = self.config.loss.pop("type")
        self.loss = loss_mapper[type_](self.config.dim, **self.config.loss)

        if self.config.nce_loss:
            self.nce_loss = loss.NCELoss(self.config.dim, self.config.nce_dim, temperature=self.config.nce_temp)

        if self.config.celltype_clf_loss:
            self.ct_clf_loss = loss.CellTypeClfLoss(self.config.dim, 164)

        self.lr = self.config.lr

    def forward(self, batch):
        mask = batch["gene_counts"] != -1
        if torch.all(mask):
            mask = None
        x = self.embedder(batch["gene_counts"])
        z = self.transformer(x, pos=batch["gene_index"], mask=mask)
        return z

    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        if not self.config.train_on_all:
            train_on = torch.isnan(batch["gene_counts"])

        y = self(batch)
        
        if not self.config.train_on_all:
            loss = self.loss(
                y[:, 1:][train_on],
                batch["gene_counts_true"][train_on],
                gene_ids=batch["gene_index"][train_on]
            )
        else:
            loss = self.loss(
                y[:, 1:], batch["gene_counts_true"], gene_ids=batch["gene_index"]
            )

        if self.config.nce_loss:
            nce_loss = self.nce_loss(y[:, 0])
            loss += nce_loss

        if self.config.celltype_clf_loss:
            ct_loss = self.ct_clf_loss(y[:, 0], batch["0/obs"][:, 3])
            loss += ct_loss

        
        self.log("train_loss", loss , sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        if not self.config.train_on_all:
            train_on = torch.isnan(batch["gene_counts"])

        y = self(batch)
        
        if not self.config.train_on_all:
            loss = self.loss(
                y[:, 1:][train_on],
                batch["gene_counts_true"][train_on],
                gene_ids=batch["gene_index"][train_on]
            )
        else:
            loss = self.loss(
                y[:, 1:], batch["gene_counts_true"], gene_ids=batch["gene_index"]
            )

        if self.config.nce_loss:
            nce_loss = self.nce_loss(y[:, 0])
            loss += nce_loss

        if self.config.celltype_clf_loss:
            ct_loss = self.ct_clf_loss(y[:, 0], batch["0/targets"])
            loss += ct_loss

        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer


class PerturbTransformer(ExpressTransformer):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        assert self.config.nce_loss == False
        assert self.config.celltype_clf_loss == False
        assert self.config.train_on_all == True
        assert isinstance(self.loss, loss.CountMSE)

        self.perturbation_indicator = nn.Parameter(torch.empty(self.config.dim).uniform_(-1, 1))
        self.validation_step_outputs = []

    def forward(self, batch):
        mask = batch["gene_counts"] != -1
        if torch.all(mask):
            mask = None

        x = self.embedder(batch["gene_counts"])

        matches = torch.where((batch["gene_index"].T == batch["0/perturbed_gene"]).T)
        assert (matches[0] == torch.arange(len(x)).to(matches[0].device)).all()
        x[torch.arange(len(x)), matches[1]] += self.perturbation_indicator

        z = self.transformer(x, pos=batch["gene_index"], mask=mask)
        return z


    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        
        loss = self.loss(
            y[:, 1:], batch["gene_counts_copy"]-batch["gene_counts_true"], gene_ids=batch["gene_index"]
        )

        self.log("train_loss", loss , sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)

        loss = self.loss(
            y[:, 1:], batch["gene_counts_copy"]-batch["gene_counts_true"], gene_ids=batch["gene_index"]
        )

        self.log("val_loss", loss, sync_dist=True)

        y = self.loss.predict(y[:, 1:], libsize=batch["gene_counts_true"].sum())
        
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
        self.log("val_deltaspearman(IC)", np.mean(delta_spearmans), sync_dist=True)

        spearmans = []
        for i in range(len(true_expr_change)):
            s = spearmanr(all_preds[i].float().numpy(), all_trues[i].float().numpy()).statistic
            spearmans.append(s)
        self.log("val_spearman(IC)", np.mean(spearmans), sync_dist=True)

        self.validation_step_outputs.clear()


    def predict_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)

        y = self(batch)
        y = self.loss.predict(y[:, 1:], libsize=batch["gene_counts_true"].sum())
        return (y, batch["gene_counts_true"], batch["gene_counts_copy"], batch["gene_index"])

class CLSTaskTransformer(ExpressTransformer):
    def __init__(
        self,
        config
    ):
        super().__init__(config)
        assert self.config.nce_loss == False
        
        if self.config.celltype_clf_loss:
            self.loss = loss.CellTypeClfLoss(self.config.dim, self.config.cls_finetune_dim)
        elif self.config.modality_prediction_loss:
            self.loss = loss.ModalityPredictionLoss(self.config.dim, self.config.cls_finetune_dim)
        else:
            raise ValueError("At least one of celltype clf loss or modality predict loss should be true")

        self.validation_step_outputs = []

        if self.config.celltype_clf_loss:
            self.micro_acc = MulticlassAccuracy(num_classes=self.config.cls_finetune_dim, average="micro")
            self.macro_acc = MulticlassAccuracy(num_classes=self.config.cls_finetune_dim, average="macro")

    def forward(self, batch):
        mask = batch["gene_counts"] != -1
        if torch.all(mask):
            mask = None
        x = self.embedder(batch["gene_counts"])
        z = self.transformer(x, pos=batch["gene_index"], mask=mask)
        return self.loss.predict(z[:, 0])


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