import torch
from logger import transformations_logger


def compress_features_pca(
    features: torch.Tensor,
    target_dim: int,
    standardize: bool = True,
) -> tuple[torch.Tensor, float]:
    if target_dim >= features.shape[1]:
        raise ValueError(
            f"target_dim={target_dim} debe ser menor que D={features.shape[1]}"
        )
    if standardize:
        features = (features - features.mean(dim=0)) / features.std(dim=0)

    P, D = features.shape
    q = min(P, D)  # rango completo real

    U, S, V = torch.pca_lowrank(features, q=q, center=not standardize)

    compressed = features @ V[:, :target_dim]

    total_variance = (S**2).sum()
    explained_variance = (S[:target_dim] ** 2).sum() / total_variance

    return compressed, explained_variance.item()


# Parallel GPU implementation. It does not estimate the SVD, so a benchmark is required.
def compress_batch_pca(
    features_batch: torch.Tensor,  # (K, P, D)
    target_dim: int,
    standardize: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    K, P, D = features_batch.shape
    if target_dim >= min(P, D):
        raise ValueError(
            f"target_dim={target_dim} debe ser menor que min(P={P}, D={D})={min(P, D)}"
        )

    mean = features_batch.mean(dim=1, keepdim=True)  # (K, 1, D)

    if standardize:
        std = features_batch.std(dim=1, keepdim=True, correction=1)
        transformed_features = (features_batch - mean) / (std + eps)
    else:
        transformed_features = features_batch - mean

    U, S, Vh = torch.linalg.svd(transformed_features, full_matrices=False)

    # Vh: (K, min(P,D), D) -> V: (K, D, min(P,D))
    V = Vh.transpose(-2, -1)

    compressed = transformed_features @ V[:, :, :target_dim]  # (K, P, target_dim)

    total_variance = (S**2).sum(dim=1)  # (K,)
    explained_variance = (S[:, :target_dim] ** 2).sum(dim=1) / total_variance  # (K,)

    return compressed, explained_variance
