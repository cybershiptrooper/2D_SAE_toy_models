import torch
import torch.nn.functional as F
import math

import random


def sample_circle_features(num_features, bias, random=False):
    if not random:
        angles = torch.linspace(0, 2 * torch.pi, num_features + 1)[:-1]
        x = torch.cos(angles)
        y = torch.sin(angles)
        return torch.stack([x, y], dim=1) + bias
    else:
        # sample from a 2D Gaussian with mean 0 and std 1
        # and make it unit norm
        features = torch.randn(num_features, 2)
        features = F.normalize(features, dim=1)
        return features + bias


def sample_embeddings(
    linear_feature_embeddings,
    multi_dim_feature_embeddings,
    linear_feature_freqs,
    multi_dim_feature_freqs,
    linear_exists_prob,
    multi_dim_exists_prob,
    dont_add_both=False,
):
    # sample one from linear features with probability linear_exists_prob
    linear_feature_embedding = torch.zeros_like(linear_feature_embeddings[0])
    linear_feature_indices = torch.zeros(linear_feature_embeddings.shape[0])
    if random.random() < linear_exists_prob:
        linear_feature_index = torch.multinomial(
            linear_feature_freqs, num_samples=1, replacement=True
        ).item()
        linear_feature_indices = torch.zeros(linear_feature_embeddings.shape[0])
        linear_feature_indices[linear_feature_index] = 1
        linear_feature_embedding = linear_feature_embeddings[linear_feature_index].clone()

    # sample one from multi-dim feature plane with probability multi_dim_exists_prob
    multi_dim_feature_embedding = torch.zeros_like(multi_dim_feature_embeddings[0])
    multi_dim_feature_indices = torch.zeros(multi_dim_feature_embeddings.shape[0])
    if random.random() < multi_dim_exists_prob:
        multi_dim_feature_index = torch.multinomial(
            multi_dim_feature_freqs, num_samples=1, replacement=True
        ).item()
        multi_dim_feature_indices = torch.zeros(multi_dim_feature_embeddings.shape[0])
        multi_dim_feature_indices[multi_dim_feature_index] = 1
        multi_dim_feature_embedding = multi_dim_feature_embeddings[
            multi_dim_feature_index
        ].clone()

    # if no features, call again
    if linear_feature_embedding.sum() == 0 and multi_dim_feature_embedding.sum() == 0:
        return sample_embeddings(
            linear_feature_embeddings,
            multi_dim_feature_embeddings,
            linear_feature_freqs,
            multi_dim_feature_freqs,
            linear_exists_prob,
            multi_dim_exists_prob,
        )
    if (
        linear_feature_embedding.abs().sum() + multi_dim_feature_embedding.abs().sum()
        == 0
    ):
        return sample_embeddings(
            linear_feature_embeddings,
            multi_dim_feature_embeddings,
            linear_feature_freqs,
            multi_dim_feature_freqs,
            linear_exists_prob,
            multi_dim_exists_prob,
        )
    if dont_add_both:
        # return the one with the highest probability or which isn't zero
        if (
            linear_feature_embedding.abs().sum() > 0
            and multi_dim_feature_embedding.abs().sum() > 0
        ):
            return {
                "x": linear_feature_embedding + multi_dim_feature_embedding,
                "label": (linear_feature_indices, multi_dim_feature_indices),
            }
        elif linear_feature_embedding.abs().sum() > 0:
            return {
                "x": linear_feature_embedding,
                "label": (linear_feature_indices, multi_dim_feature_indices),
            }
        else:
            return {
                "x": multi_dim_feature_embedding,
                "label": (linear_feature_indices, multi_dim_feature_indices),
            }
    return {
        "x": linear_feature_embedding + multi_dim_feature_embedding,
        "label": (linear_feature_indices, multi_dim_feature_indices),
    }


def sample_embeddings_independently(
    linear_feature_embeddings,
    multi_dim_feature_embeddings,
    linear_feature_freqs,
    multi_dim_feature_freqs,
    max_linear_features=-1,  # New parameter for max linear features
    max_multi_dim_features=-1,  # New parameter for max multi-dimensional features
):
    if max_linear_features == -1:
        # sample all linear features
        max_linear_features = linear_feature_embeddings.shape[0]
        print(f"Sampling {max_linear_features} linear features")
    if max_multi_dim_features == -1:
        # sample all multi-dim features
        max_multi_dim_features = multi_dim_feature_embeddings.shape[0]
        print(f"Sampling {max_multi_dim_features} multi-dimensional features")
    # Initialize embeddings and indices
    linear_feature_embedding = torch.zeros_like(linear_feature_embeddings[0])
    multi_dim_feature_embedding = torch.zeros_like(multi_dim_feature_embeddings[0])
    linear_feature_indices = torch.zeros(linear_feature_embeddings.shape[0])
    multi_dim_feature_indices = torch.zeros(multi_dim_feature_embeddings.shape[0])

    # Create a random permutation of indices for linear features
    linear_indices = torch.randperm(linear_feature_embeddings.shape[0])
    sampled_linear_count = 0  # Counter for sampled linear features
    for i in linear_indices:
        if (
            random.random() < linear_feature_freqs[i]
            and sampled_linear_count < max_linear_features
        ):
            linear_feature_embedding += linear_feature_embeddings[i].clone()
            linear_feature_indices[i] = 1
            sampled_linear_count += 1  # Increment the counter

    # Create a random permutation of indices for multi-dimensional features
    multi_dim_indices = torch.randperm(multi_dim_feature_embeddings.shape[0])
    sampled_multi_dim_count = 0  # Counter for sampled multi-dimensional features
    for i in multi_dim_indices:
        if (
            random.random() < multi_dim_feature_freqs[i]
            and sampled_multi_dim_count < max_multi_dim_features
        ):
            multi_dim_feature_embedding += multi_dim_feature_embeddings[i].clone()
            multi_dim_feature_indices[i] = 1
            sampled_multi_dim_count += 1  # Increment the counter

    return {
        "x": linear_feature_embedding + multi_dim_feature_embedding,
        "label": (linear_feature_indices, multi_dim_feature_indices),
    }


def generate_feature_embeddings(
    num_multi_dim_features: int,
    d_in: int,
    num_linear_features: int,
    orthogonalize_coefficient: float = 0.0,
    noise_dim: int = 1,
    noise_scale: float = 0.005,
    feature_scale: float = 1.0,
    radial_noise: float = 0.005,
    bias_scale: float = 1.0,
    add_bias: bool = False,
):
    angles = torch.arange(0, 2 * math.pi, (2 * math.pi / num_multi_dim_features))
    x = torch.cos(angles)
    y = torch.sin(angles)
    # add random noise to the z axis of noise = sin(x)
    noise = torch.randn(num_multi_dim_features, noise_dim) * noise_scale
    # z = torch.sin(20 * x) * noise_scale
    # z = torch.zeros_like(x)
    # multi_dim_feature_embeddings = torch.cat(
    #     [x.unsqueeze(1), y.unsqueeze(1), z.unsqueeze(1)], dim=1
    # )

    # concatenate circle and noise
    multi_dim_feature_embeddings = torch.cat(
        [x.unsqueeze(1), y.unsqueeze(1), noise], dim=1
    )

    # transform this to a random 2d subspace of 10 dimensions
    random_matrix = torch.randn(d_in, noise_dim + 2)
    Q, _ = torch.linalg.qr(random_matrix)  # Q is now a 10x2 orthonormal matrix

    multi_dim_feature_embeddings = torch.einsum(
        "ij,kj->ki", Q, multi_dim_feature_embeddings
    )

    projection_matrix = Q @ Q.T  # This is a (10, 10) matrix

    # Generate linear feature embeddings
    linear_feature_embeddings = torch.randn(num_linear_features, d_in)

    # Project the embeddings onto the 2D subspace
    projected_embeddings = linear_feature_embeddings @ projection_matrix

    # Subtract the projection to get the orthogonal part
    linear_feature_embeddings = (
        linear_feature_embeddings - projected_embeddings * orthogonalize_coefficient
    )
    linear_feature_embeddings /= linear_feature_embeddings.norm(dim=1, keepdim=True)

    # assert that multi_dim_feature_embeddings' pca is 2
    assert (
        torch.linalg.matrix_rank(multi_dim_feature_embeddings) == noise_dim + 2
    ), torch.linalg.matrix_rank(multi_dim_feature_embeddings)

    # rescale the embeddings
    linear_feature_embeddings *= feature_scale
    radial_noise = torch.randn(d_in) * radial_noise
    linear_feature_embeddings += radial_noise

    multi_dim_feature_embeddings *= feature_scale
    radial_noise = torch.randn(d_in) * radial_noise
    multi_dim_feature_embeddings += radial_noise

    if add_bias:
        bias = torch.randn(d_in)
        bias /= bias.norm()
        bias *= bias_scale
        linear_feature_embeddings += bias
        multi_dim_feature_embeddings += bias
    else:
        bias = torch.zeros(d_in)

    return {
        "linear_feature_embeddings": linear_feature_embeddings,
        "multi_dim_feature_embeddings": multi_dim_feature_embeddings,
        "bias": bias,
    }
