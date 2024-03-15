import torch
from torch import optim, nn
import torch.nn.functional as F
import lightning.pytorch as pl
from bio_attention.attention import TransformerEncoder
from bio_attention.embed import DiscreteEmbedding, ContinuousEmbedding


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
        discrete_input=False,
        n_discrete_tokens=10,
        dim=64,
        depth=8,
        dropout=0.15,
        n_genes=19331,
        lr=1e-4,
    ):
        super().__init__()
        if discrete_input:
            self.embedder = DiscreteEmbedding(n_discrete_tokens, dim, cls=True)
        else:
            self.embedder = ContinuousEmbedding(dim, cls=True)

        self.transformer = TransformerEncoder(
            depth=depth,
            dim=dim,
            nh=8,
            attentiontype="vanilla",
            attention_args={"dropout": dropout},
            plugintype="learned",
            plugin_args={"dim": dim, "max_seq_len": n_genes},
            only_apply_plugin_at_first=False,
            dropout=dropout,
            glu_ff=True,
            activation="gelu",
        )

        self.output_head = nn.Linear(dim, n_discrete_tokens)
        self.lr = lr

    def forward(self, batch):
        mask = batch["gene_counts"] != -1
        x = self.embedder(batch["gene_counts"])
        z = self.transformer(x, pos=batch["gene_index"], mask=mask)
        return self.output_head(z)

    def training_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)
        train_on = torch.isnan(batch["gene_counts"])

        y = self(batch)
        y = y[:, 1:]

        loss = F.cross_entropy(y[train_on], batch["gene_counts_true"][train_on])

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch["gene_counts"] = batch["gene_counts"].to(self.dtype)
        train_on = torch.isnan(batch["gene_counts"])

        y = self(batch)
        y = y[:, 1:]

        loss = F.cross_entropy(y[train_on], batch["gene_counts_true"][train_on])

        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        return optimizer
