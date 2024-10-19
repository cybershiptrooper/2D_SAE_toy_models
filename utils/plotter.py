import torch
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


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
