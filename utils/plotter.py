import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def desc_embeddings(linear_feature_embeddings, multi_dim_feature_embeddings):
    dot_products = torch.matmul(multi_dim_feature_embeddings, linear_feature_embeddings.T)
    plt.imshow(dot_products.detach().numpy(), cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Dot Product")
    plt.xlabel("Linear Embeddings")
    plt.ylabel("Multi-dim Embeddings")
    plt.show()
    # show dot product between linear_feature_embeddings and linear_feature_embeddings
    dot_products = torch.matmul(linear_feature_embeddings, linear_feature_embeddings.T)
    plt.imshow(dot_products.detach().numpy(), cmap="coolwarm", aspect="auto")
    plt.colorbar(label="Dot Product")
    plt.xlabel("Linear Embeddings")
    plt.ylabel("Linear Embeddings")
    plt.show()


def plot_cosine_similarity(cosine_similarity, x_labels, y_labels):
    # Create a heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=cosine_similarity,
            colorscale="RdBu",
            zmin=-1,  # Set minimum value for better color scaling
            zmax=1,  # Set maximum value for better color scaling
        )
    )

    # Update the layout
    fig.update_layout(
        xaxis_title=x_labels,
        yaxis_title=y_labels,
        width=500,
        height=500,
    )
    fig.update_yaxes(autorange="reversed")
    fig.show()


def plot_latent_activations(
    m_sae: torch.nn.Module,
    linear_feature_embeddings: torch.Tensor,
    multi_dim_feature_embeddings: torch.Tensor,
    use_linear=True,
):
    features = torch.cat([linear_feature_embeddings, multi_dim_feature_embeddings], dim=0)
    latents = m_sae.encode(features)[0 if use_linear else 1].detach().numpy()

    fig = make_subplots(rows=1, cols=1)

    heatmap = go.Heatmap(z=latents, colorscale="Viridis")
    fig.add_trace(heatmap)

    fig.add_shape(
        type="line",
        x0=0 - 0.5,
        x1=latents.shape[1] - 0.5,
        y0=linear_feature_embeddings.shape[0] - 0.5,
        y1=linear_feature_embeddings.shape[0] - 0.5,
        line=dict(color="red", width=2),
    )

    fig.update_xaxes(title_text="Latent Space", tickmode="linear", tick0=0, dtick=1)
    fig.update_yaxes(showticklabels=False)

    fig.add_annotation(
        x=-0.0,
        y=linear_feature_embeddings.shape[0] / 2,
        text="Linear<br>Features",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        xref="paper",
        yref="y",
    )

    fig.add_annotation(
        x=-0.0,
        y=linear_feature_embeddings.shape[0] + multi_dim_feature_embeddings.shape[0] / 2,
        text="Multi-dim<br>Features",
        showarrow=False,
        xanchor="right",
        yanchor="middle",
        xref="paper",
        yref="y",
    )

    fig.update_layout(
        # title=f"{'Linear' if use_linear else 'Multidimensional'} Latent Activations",
        width=800,
        height=600,
    )
    fig.update_yaxes(autorange="reversed")

    fig.show()
