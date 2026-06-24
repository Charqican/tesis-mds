import torch
from logger import transformations_logger


def compress_features_pca(
    features: torch.Tensor,  # (P, D) features de UN objeto
    target_dim: int,
) -> tuple[torch.Tensor, float]:
    """
    Args:
        features: (P, D) features por punto de un objeto
        target_dim: dimension de salida deseada

    Returns:
        compressed: (P, target_dim) features proyectadas
        explained_variance_ratio: fraccion promedio de varianza explicada
    """

    U, S, V = torch.pca_lowrank(features, q=target_dim, center=True)

    compressed = features @ V[:, :target_dim]  # (P, target_dim)

    total_variance = (S**2).sum()
    explained_variance = (S[:target_dim] ** 2).sum() / total_variance

    transformations_logger.debug(
        f"PCA | target_dim={target_dim} "
        f"explained_variance={explained_variance.item():.4f}"
    )

    return compressed, explained_variance.item()


# Parallel GPU implementation. It does not estimate the SVD, so a benchmark is required.
def compress_batch_pca(
    features_batch: torch.Tensor,  # (K, P, D)
    target_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    mean = features_batch.mean(dim=1, keepdim=True)  # (K, 1, D)
    centered = features_batch - mean  # (K, P, D)

    U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
    # S: (K, min(P,D))

    V = Vh.transpose(-2, -1)  # (K, D, min(P,D))
    compressed = centered @ V[:, :, :target_dim]  # (K, P, target_dim)

    total_variance = (S**2).sum(dim=1)  # (K,)
    explained_variance = (S[:, :target_dim] ** 2).sum(dim=1) / total_variance  # (K,)

    return compressed, explained_variance
