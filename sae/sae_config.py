from dataclasses import dataclass

import torch.nn.functional as F


@dataclass
class MultiDimSAEConfig:
    num_multi_dim_features: int
    gate_threshold: float = 0.05


@dataclass
class SAEConfig:
    input_dim: int
    latent_dim: int
    l1_weight: float = 0.05
    activation_fn: callable = F.relu
    learning_rate: float = 0.01
    multi_dim_config: MultiDimSAEConfig = None
    gradient_clipping: float = 1.0
