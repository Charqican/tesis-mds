import torch
from metrics.utils import get_top_L_similar_index


def compute_symmetry_invariance(
    points: torch.Tensor,  # (1000, 3) - Nube de puntos submuestreada
    features: torch.Tensor,  # (1000, D) - Features correspondientes
    normal: torch.Tensor,  # (3,) - Normal del plano
    midpoint: torch.Tensor,  # (3,) - Punto medio del plano
) -> float:
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

    # calculate point similarity
    distance = torch.mean(torch.sum(torch.abs(features - features_reflected), dim=1))

    return distance.item()
