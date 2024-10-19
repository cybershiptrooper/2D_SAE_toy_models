from utils.embeddings import centered_embeddings
import torch


def test_centered_embeddings():
    num_linear_features = 100
    d_in = 20
    num_multi_dim_features = 100

    linear_feature_embeddings = torch.randn(num_linear_features, d_in)
    multi_dim_feature_embeddings = torch.randn(num_multi_dim_features, d_in)
    common_bias = torch.randn(d_in)

    original_linear_feature_embeddings = linear_feature_embeddings.clone()
    original_multi_dim_feature_embeddings = multi_dim_feature_embeddings.clone()
    original_common_bias = common_bias.clone()

    with centered_embeddings(
        common_bias, linear_feature_embeddings, multi_dim_feature_embeddings
    ):
        assert linear_feature_embeddings.norm(dim=1).mean().item() == 1.0
        assert multi_dim_feature_embeddings.norm(dim=1).mean().item() == 1.0

    assert torch.equal(linear_feature_embeddings, original_linear_feature_embeddings)
    assert torch.equal(
        multi_dim_feature_embeddings, original_multi_dim_feature_embeddings
    )
    assert torch.equal(common_bias, original_common_bias)
