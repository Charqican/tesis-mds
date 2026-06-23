import torch
from logger import metrics_logger


def get_top_L_similar_index(
    query: torch.Tensor,  # (N, D)
    vec_set: torch.Tensor,  # (M, D)
    L: int = 1,
    mask_self_similarity: bool = False,
) -> torch.Tensor:
    """ """
    device = query.device
    N, M = query.shape[0], vec_set.shape[0]

    distances = torch.cdist(query, vec_set, p=2)

    if mask_self_similarity:
        diag_indices = torch.arange(min(N, M), device=device)
        distances[diag_indices, diag_indices] = float("inf")

    _, top_L_indices = torch.topk(distances, L, dim=1, largest=False)

    return top_L_indices
