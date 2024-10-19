import torch
import numpy as np


def compute_similarities(
    m_sae, linear_feature_embeddings, multi_dim_feature_embeddings, use_linear_SAE=True
):
    """
    Compute the similarity between the original feature embeddings and the reconstructed feature embeddings.
    """
    linear_similarities = []
    multi_similarities = []
    with torch.no_grad():
        for embeddings, similarities in [
            (linear_feature_embeddings, linear_similarities),
            (multi_dim_feature_embeddings, multi_similarities),
        ]:
            for embedding in embeddings:
                latent_for_embedding = m_sae.encode(embedding.unsqueeze(0))
                if use_linear_SAE:
                    reconstructed_embedding = m_sae.decode(
                        (
                            latent_for_embedding[0],
                            torch.zeros_like(latent_for_embedding[1]),
                        )
                    )
                else:
                    reconstructed_embedding = m_sae.decode(
                        (
                            torch.zeros_like(latent_for_embedding[0]),
                            latent_for_embedding[1],
                        )
                    )
                similarities.append(
                    (embedding - reconstructed_embedding.squeeze(0)).norm().item()
                )

    return np.array(linear_similarities), np.array(multi_similarities)
