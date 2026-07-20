import torch


def _orthonormal_basis(n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construye una base ortonormal (u, v) del plano perpendicular a `n`,

    param n: normal del plano de simetria (3,), no necesita venir normalizado
    return: (u, v) : Torch(3), unitarios y perpendiculares entre si y a n
    """
    n = n / n.norm()

    # eje canonico menos paralelo a n, para evitar degeneracion en la
    # proyeccion (si n ya es casi paralelo a [1,0,0], se usa [0,1,0])
    reference = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype)
    if torch.abs(torch.dot(n, reference)) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype)

    u = reference - torch.dot(reference, n) * n
    u = u / u.norm()
    v = torch.cross(n, u, dim=-1)

    return u, v


def sample_random_viewpoint(points: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    """
    Punto aleatorio uniforme sobre la esfera de radio `radius`, usado
    usado como viewpoint
    param points: nube de puntos (N, 3)
    param radius: radio de la esfera sobre la que se samplea el viewpoint
    return: viewpoint (3,)
    """
    direction = torch.randn(3, device=points.device, dtype=points.dtype)
    direction = direction / direction.norm()
    return radius * direction


def sample_equator_point(
    symmetry_plane: tuple[torch.Tensor, torch.Tensor], angle: float, radius: float = 1.0
) -> torch.Tensor:
    """
    Punto sobre el ecuador, circulo maximo contenido en el plano de
    simetria.

    param symmetry_plane: normal del plano de simetria
    param angle: angulo en [0, 2*pi)
    param radius: radio de la esfera
    return: viewpoint (3,) sobre el ecuador
    """
    n_vec, plane_pt = symmetry_plane
    u, v = _orthonormal_basis(n_vec)
    angle_t = torch.as_tensor(angle, device=u.device, dtype=u.dtype)
    return plane_pt + radius * (torch.cos(angle_t) * u + torch.sin(angle_t) * v)


def sample_meridian_point(
    symmetry_plane: tuple[torch.Tensor, torch.Tensor], angle: float, radius: float = 1.0
) -> torch.Tensor:
    """
    Punto sobre el meridiano principal: circulo maximo que contiene el
    eje normal `symmetry_plane` y el punto del ecuador en angle=0.

    param symmetry_plane: normal del plano de simetria
    param angle: latitud en [-pi/2, pi/2]
    param radius: radio de la esfera
    return: viewpoint (3,) sobre el meridiano principal
    """
    n_vec, plane_pt = symmetry_plane
    n = n_vec / n_vec.norm()
    u, _ = _orthonormal_basis(n_vec)
    lat = torch.as_tensor(angle, device=u.device, dtype=u.dtype)
    return plane_pt + radius * (torch.cos(lat) * u + torch.sin(lat) * n)


def sample_transverse_meridian_point(
    symmetry_plane: tuple[torch.Tensor, torch.Tensor], angle: float, radius: float = 1.0
) -> torch.Tensor:
    """
    Punto sobre el meridiano transversal, circulo maximo que tambien
    contiene el eje normal `symmetry_plane`, pero usando la direccion
    perpendicular v = n x u en vez de u.

    :param symmetry_plane: normal del plano de simetria y punto en el plano
    :param angle: latitud en [-pi/2, pi/2]
    :param radius: radio de la esfera
    :return: viewpoint (3,) sobre el meridiano transversal
    """
    n_vec, plane_pt = symmetry_plane
    n = n_vec / n_vec.norm()
    _, v = _orthonormal_basis(n_vec)
    lat = torch.as_tensor(angle, device=v.device, dtype=v.dtype)
    return plane_pt + radius * (torch.cos(lat) * v + torch.sin(lat) * n)


def separate_point_cloud(
    points: torch.Tensor, center: torch.Tensor, k: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Separa `points` removiendo el k% de puntos mas lejanos a `center`.

    param points: nube de puntos completa (N, 3)
    param center: viewpoint (3,)
    param k: fraccion en [0, 1] de puntos mas lejanos a remover
    return: (partial_points, mask)
        partial_points: (M, 3) con M = round(N * (1 - k))
        mask: booleana (N,), True = punto conservado
    """
    distances = (points - center).norm(dim=-1)  # (N,)
    n_points = points.shape[0]
    n_keep = int(round(n_points * (1.0 - k)))

    order = torch.argsort(distances)  # ascendente: mas cercano primero
    keep_idx = order[:n_keep]

    mask = torch.zeros(n_points, dtype=torch.bool, device=points.device)
    mask[keep_idx] = True

    return points[mask], mask
