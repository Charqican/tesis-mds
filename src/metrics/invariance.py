import torch
from metrics.utils import get_top_L_similar_index


def _symm_invariance(
    points: torch.Tensor,  # (N, 3) - Nube de puntos submuestreada
    features: torch.Tensor,  # (N, D) - Features correspondientes
    normal: torch.Tensor,  # (3,) - Normal del plano
    midpoint: torch.Tensor,  # (3,) - Punto medio del plano
) -> tuple[torch.Tensor, torch.Tensor]:

    device = points.device

    # reflect points using the plane of symmetry
    vector_to_points = points - midpoint
    projection_scalar = torch.sum(vector_to_points * normal, dim=1, keepdim=True)
    projection_vector = projection_scalar * normal
    points_reflected = points - 2 * projection_vector

    idx = get_top_L_similar_index(
        query=points, vec_set=points_reflected, L=1, mask_self_similarity=False
    )
    # pair points
    features_reflected = features[idx.squeeze(1)]

    return features, features_reflected


def compute_symmetry_invariance_pointwise(
    points: torch.Tensor,  # (N, 3) - Nube de puntos submuestreada
    features: torch.Tensor,  # (N, D) - Features correspondientes
    normal: torch.Tensor,  # (3,) - Normal del plano
    midpoint: torch.Tensor,  # (3,) - Punto medio del plano
) -> torch.Tensor:
    features, features_reflected = _symm_invariance(points, features, normal, midpoint)

    return torch.sum(torch.abs(features - features_reflected), dim=1)


def compute_symmetry_invariance(
    points: torch.Tensor,  # (N, 3) - Nube de puntos submuestreada
    features: torch.Tensor,  # (N, D) - Features correspondientes
    normal: torch.Tensor,  # (3,) - Normal del plano
    midpoint: torch.Tensor,  # (3,) - Punto medio del plano
) -> float:
    # calculate point similarity
    distances = compute_symmetry_invariance_pointwise(
        points, features, normal, midpoint
    )
    return torch.mean(distances).item()


def compute_symmetry_invariance_object(
    points: torch.Tensor,  # (N, 3)
    features: torch.Tensor,  # (N, D)
    planes: list[tuple[torch.Tensor, torch.Tensor]],  # [(normal, midpoint), ...]
) -> float:
    """
    Distancia L1 de invarianza promediada sobre todos los planos
    de simetria de un objeto.
    """
    distances = [
        compute_symmetry_invariance(points, features, normal, midpoint)
        for normal, midpoint in planes
    ]
    return sum(distances) / len(distances)
