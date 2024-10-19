import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import plotly.express as px
import pandas as pd
from sae.sae_config import SAEConfig


class SAE(nn.Module):
    def __init__(self, config: SAEConfig):
        super(SAE, self).__init__()
        self.config = config
        self.W_enc = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(config.input_dim, config.latent_dim))
        )
        self.b_enc = nn.Parameter(torch.zeros(config.latent_dim))
        self.W_dec = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(config.latent_dim, config.input_dim))
        )
        self.b_dec = nn.Parameter(torch.zeros(config.input_dim))
        self.activation_fn = config.activation_fn
        self.l1_weight = config.l1_weight
        self.learning_rate = config.learning_rate

    def encode(self, x):
        encoded = torch.einsum("bi,ih->bh", x - self.b_dec, self.W_enc) + self.b_enc
        return self.activation_fn(encoded)

    def decode(self, x):
        return torch.einsum("bh,hi->bi", x, self.W_dec) + self.b_dec

    def compute_loss(self, output, latent, dataset):
        return (
            F.mse_loss(output, dataset) + self.l1_weight * latent.abs().sum(dim=-1).mean()
        )

    def eval_step(self, dataset):
        with torch.no_grad():
            latents = self.encode(dataset)
            outputs = self.decode(latents)
            l1_sparsity = latents.abs().sum(dim=1).mean().item()
            l0_norm = (latents > 0).float().sum(dim=1).mean().item()
            return {
                "l1_sparsity": l1_sparsity,
                "l0_norm": l0_norm,
                "loss": F.mse_loss(outputs, dataset).item(),
            }

    def normalize_weights(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def clip_gradients(self):
        with torch.no_grad():
            for param in self.parameters():
                if param.grad is not None:
                    param.grad.data = param.grad.data.clamp(
                        -self.config.gradient_clipping, self.config.gradient_clipping
                    )

    def train_model(self, dataset, num_epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        logs = []
        for epoch in tqdm(range(num_epochs)):
            optimizer.zero_grad()
            latents = self.encode(dataset)
            outputs = self.decode(latents)
            loss = self.compute_loss(outputs, latents, dataset)
            loss.backward()
            self.clip_gradients()
            optimizer.step()
            # normalize the decoder and encoder weights
            self.normalize_weights()
            logs.append(self.eval_step(dataset))

        self.logs = pd.DataFrame(logs)
        return self

    def plot_losses(self):
        self.plot("loss")
        self.plot("l1_sparsity")
        self.plot("l0_norm")

    def plot(self, key: str):
        fig = px.line(self.logs, y=key, x=self.logs.index)
        fig.show()
