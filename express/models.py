import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import DiscreteEmbedding, ContinuousEmbedding
from express import loss
import yaml


class EmbeddingGater(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(dim).uniform_(-1, 1))

    def forward(self, x):
        return F.tanh(x) * self.embedding


class EmbeddingPseudoQuantizer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(in_dim, out_dim).uniform_(-1, 1))

    def forward(self, x):
        return F.softmax(x, dim=-1) @ self.embedding


class ExpressTransformer(pl.LightningModule):
    def __init__(
        self,
        config_path
    ):
        super().__init__()

        with open(config_path) as f:
            self.config = yaml.load(f, Loader=yaml.Loader)

        if self.config.discrete_input:
            self.embedder = DiscreteEmbedding(self.config.n_discrete_tokens, self.config.dim, cls=True)
        else:
            self.embedder = ContinuousEmbedding(self.config.dim, cls=True)

        self.transformer = TransformerEncoder(
            depth=self.config.depth,
            dim=self.config.dim,
            nh=8,
            attentiontype="vanilla",
            attention_args={"dropout": self.config.dropout},
            plugintype="learned",
            plugin_args={"dim": self.config.dim, "max_seq_len": self.config.n_genes},
            only_apply_plugin_at_first=True,
            dropout=self.config.dropout,
            glu_ff=True,
            activation="gelu",
        )

        self.output_head = nn.Linear(self.config.dim, self.config.n_discrete_tokens)
        
        
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
        x = self.embedder(batch["gene_counts"])
        z = self.transformer(x, pos=batch["gene_index"], mask=mask)
        return self.output_head(z)

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
            ct_loss = self.ct_clf_loss(y[:, 0], batch["0/obs"][:, 3])
            loss += ct_loss

        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
