class centered_embeddings:
    """
    A context manager which can be used to center and normalize embeddings.
    analogous to
    with torch.no_grad():
        # do stuff
    """

    def __init__(self, bias, linear_feature_embeddings, multi_dim_feature_embeddings):
        self.bias = bias
        self.linear_feature_embeddings = linear_feature_embeddings
        self.multi_dim_feature_embeddings = multi_dim_feature_embeddings
        self._original_linear_feature_embeddings = linear_feature_embeddings.clone()
        self._original_multi_dim_feature_embeddings = multi_dim_feature_embeddings.clone()
        self.center_embeddings = False  # Flag to track the centering state

    def _apply_centering(self):
        """Apply centering and normalization to embeddings."""
        self.linear_feature_embeddings -= self.bias
        self.multi_dim_feature_embeddings -= self.bias
        self.linear_feature_embeddings /= self.linear_feature_embeddings.norm(
            dim=1, keepdim=True
        )
        self.multi_dim_feature_embeddings /= self.multi_dim_feature_embeddings.norm(
            dim=1, keepdim=True
        )

    def _reset_embeddings(self):
        """Restore embeddings to their original state."""
        self.linear_feature_embeddings.copy_(self._original_linear_feature_embeddings)
        self.multi_dim_feature_embeddings.copy_(
            self._original_multi_dim_feature_embeddings
        )

    def __enter__(self):
        # Automatically center embeddings when entering the context
        self._apply_centering()
        return self.linear_feature_embeddings, self.multi_dim_feature_embeddings

    def __exit__(self, exc_type, exc_value, traceback):
        # Reset the embeddings to their original state when exiting the context
        self._reset_embeddings()
