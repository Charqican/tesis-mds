from dataclasses import dataclass

import numpy as np

from .geometry_utils import nn_residuals
from .registrate import RegistrationResult


@dataclass(frozen=True)
class SanityMetrics:
    """Métricas de calidad del registro."""

    # Primaria: escena -> mesh (robusta a oclusión)
    fwd_median: float
    fwd_mean: float
    fwd_p90: float
    fwd_max: float
    fwd_n: int

    # Secundaria: mesh -> escena (inflada por oclusión)
    bwd_median: float
    bwd_mean: float
    bwd_p90: float
    bwd_max: float
    bwd_n: int

    def __str__(self) -> str:
        lines = [
            f"[sanity] PRIMARY  scene->mesh (observed pts -> posed model), n={self.fwd_n}:",
            f"         median {self.fwd_median:.2f} | mean {self.fwd_mean:.2f} | "
            f"p90 {self.fwd_p90:.2f} | max {self.fwd_max:.2f}  [mm]",
            f"         -> pose is good if median is a few mm. Insensitive to occlusion.",
            f"[sanity] SECONDARY mesh->scene (full model -> scene), n={self.bwd_n}:",
            f"         median {self.bwd_median:.2f} | mean {self.bwd_mean:.2f} | "
            f"p90 {self.bwd_p90:.2f} | max {self.bwd_max:.2f}  [mm]",
            f"         -> inflated by occlusion & full-surface sampling; the tail is expected.",
        ]
        return "\n".join(lines)


def compute_sanity_metrics(result: RegistrationResult) -> SanityMetrics:
    """Calcula métricas de calidad del registro."""

    # Referencia para métrica secundaria: object_pts si existe, si no scene_pts completa
    ref = result.object_pts if result.object_pts is not None else result.scene_pts

    # Métrica primaria: escena -> mesh (solo si tenemos puntos del objeto)
    fwd_median = fwd_mean = fwd_p90 = fwd_max = 0.0
    fwd_n = 0

    if result.object_pts is not None and len(result.object_pts) > 0:
        res_fwd = nn_residuals(result.object_pts, result.posed_mesh_pts)
        fwd_median = float(np.median(res_fwd))
        fwd_mean = float(res_fwd.mean())
        fwd_p90 = float(np.percentile(res_fwd, 90))
        fwd_max = float(res_fwd.max())
        fwd_n = len(res_fwd)

    # Métrica secundaria: mesh -> escena
    res_bwd = nn_residuals(result.posed_mesh_pts, ref)

    return SanityMetrics(
        fwd_median=fwd_median,
        fwd_mean=fwd_mean,
        fwd_p90=fwd_p90,
        fwd_max=fwd_max,
        fwd_n=fwd_n,
        bwd_median=float(np.median(res_bwd)),
        bwd_mean=float(res_bwd.mean()),
        bwd_p90=float(np.percentile(res_bwd, 90)),
        bwd_max=float(res_bwd.max()),
        bwd_n=len(res_bwd),
    )
