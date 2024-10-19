import torch
from torch import nn
from torch.nn import functional as F

from sae.sae import SAE
from sae.sae_config import SAEConfig


class MultiDimSAE(SAE):
    def __init__(self, config: SAEConfig):
        super().__init__(config)
        assert config.multi_dim_config is not None, "MultiDimSAEConfig is required"
        self.config = config
        self.m_latent = config.multi_dim_config.num_multi_dim_features
        self.input_dim = config.input_dim

        self.b_enc_multi_dim = nn.Parameter(torch.zeros(self.m_latent))
        self.W_enc_multi_dim = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.input_dim, self.m_latent))
        )
        self.W_dec_multi_dim = nn.Parameter(
            nn.init.kaiming_uniform_(torch.empty(self.m_latent, self.input_dim))
        )

    def encode(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_linear = super().encode(x)
        mapped_multi_dim = (
            torch.einsum("bi,ih->bh", x - self.b_dec, self.W_enc_multi_dim)
            + self.b_enc_multi_dim
        )
        # apply activation fn (x1^2 + x2^2)^0.5 (where x1, x2 are adjacent feature)
        assert (
            mapped_multi_dim.shape[1] % 2 == 0
        ), "Number of multi-dim features must be even"
        half_len = mapped_multi_dim.shape[1] // 2
        x1_features = mapped_multi_dim[:, :half_len]
        x2_features = mapped_multi_dim[:, half_len:]
        mags = (x1_features**2 + x2_features**2 + 1e-6) ** 0.5
        gate = (mags > self.config.multi_dim_config.gate_threshold).float()
        # make the gate for each feature
        gate = torch.cat([gate, gate], dim=1)
        # apply the gate
        encoded_multi_dim = gate * mapped_multi_dim
        return (encoded_linear, encoded_multi_dim)

    def decode(self, x):
        encoded_linear, encoded_multi_dim = x
        decoded_linear = super().decode(encoded_linear)
        decoded_multi_dim = torch.einsum(
            "bh,hi->bi", encoded_multi_dim, self.W_dec_multi_dim
        )
        return decoded_linear + decoded_multi_dim

    def compute_loss(self, output, latent: tuple[torch.Tensor, torch.Tensor], dataset):
        encoded_linear, encoded_multi_dim = latent
        output = self.decode(latent)

        # Check for NaNs
        assert not torch.isnan(output).any(), "Output contains NaNs"
        assert not torch.isnan(encoded_linear).any(), "Encoded linear contains NaNs"

        linear_sparsity = encoded_linear.abs().sum(dim=1).mean()

        # multi dim sparsity is sqrt(z_1^2 + z_2^2)
        half_len = encoded_multi_dim.shape[1] // 2
        x1 = encoded_multi_dim[:, :half_len]
        x2 = encoded_multi_dim[:, half_len:]

        # Check for NaNs
        assert not torch.isnan(x1).any(), "x1 contains NaNs"
        assert not torch.isnan(x2).any(), "x2 contains NaNs"

        multi_dim_sparsity = (x1**2 + x2**2 + 1e-6).sqrt().sum(dim=1).mean()

        loss = F.mse_loss(output, dataset) + self.l1_weight * (
            linear_sparsity + multi_dim_sparsity
        )
        assert not torch.isnan(
            loss
        ).any(), f"Loss contains NaNs: {loss}, linear_sparsity: {linear_sparsity}, multi_dim_sparsity: {multi_dim_sparsity}"
        return loss

    def eval_step(self, dataset):
        with torch.no_grad():
            latents = self.encode(dataset)
            outputs = self.decode(latents)
            l1_sparsity = (
                latents[0].abs().sum(dim=1).mean().item()
                + latents[1].abs().sum(dim=1).mean().item()
            )
            l0_norm = (latents[0] > 0).float().sum(dim=1).mean().item() + (
                latents[1] > 0
            ).float().sum(dim=1).mean().item()
            return {
                "l1_sparsity": l1_sparsity,
                "l0_norm": l0_norm,
                "loss": F.mse_loss(outputs, dataset).item(),
            }

    def normalize_weights(self):
        with torch.no_grad():
            self.W_dec_multi_dim.data = F.normalize(self.W_dec_multi_dim.data, dim=1)
            super().normalize_weights()
