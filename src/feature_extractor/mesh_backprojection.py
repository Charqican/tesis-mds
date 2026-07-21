import numpy as np
import torch
from collections import defaultdict
from pytorch3d.renderer import (
    DirectionalLights,
    FoVPerspectiveCameras,
    HardFlatShader,
    MeshRasterizer,
    MeshRendererWithFragments,
    RasterizationSettings,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes

from feature_extractor.config import FeatureConfig
from logger import backprojection_logger
from model_wrappers import (
    DINOWrapper,
)  # ajustar import segun ubicacion real

# rot_aug=(0,1,2,3,4) ---> (0) VS (0, 180) VS (0, 90, 270) VS (0, 90, 180, 270) VS (0, 0, 0, 0)
rotation_maps = {
    0: [0],  # sin rotacion (solo 0 grados)
    1: [0, 2],  # 0 y 180 grados
    2: [0, 1, 3],  # 0, 90, 270 grados
    3: [0, 1, 2, 3],  # las cuatro rotaciones
    4: [0, 0, 0, 0],  # cuatro copias del original
}


def find_adjacent_faces(faces: torch.Tensor) -> torch.Tensor:  # (F, max_adjacent_faces)
    """
    :param faces: caras del mesh (F, 3)
    :return: tensor (F, max_adjacent_faces) con indices de caras
        adyacentes; -1 en las posiciones sin cara adyacente (padding,
        ya que no todas las caras tienen el mismo numero de vecinas)
    """
    device = faces.device
    faces_list = faces.tolist()

    # mapeo de arista -> caras que la contienen
    edge_to_faces = defaultdict(list)
    for i, face in enumerate(faces_list):
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            edge_to_faces[edge].append(i)

    adjacent_faces = []
    max_adjacent_faces = 0
    for i, face in enumerate(faces_list):
        neighbors = set()
        for j in range(3):
            edge = tuple(sorted((face[j], face[(j + 1) % 3])))
            for neighbor_face in edge_to_faces[edge]:
                if neighbor_face != i:
                    neighbors.add(neighbor_face)
        adjacent_faces.append(list(neighbors))
        max_adjacent_faces = max(max_adjacent_faces, len(neighbors))

    ret = (
        torch.zeros(
            len(faces_list), max_adjacent_faces, dtype=torch.long, device=device
        )
        - 1
    )
    for i, neighbors in enumerate(adjacent_faces):
        ret[i, : len(neighbors)] = torch.tensor(
            neighbors, dtype=torch.long, device=device
        )

    return ret


def check_visible_vertices_optimized(
    pix_to_face: torch.Tensor,  # (V, H, W, faces_per_pixel)
    mesh: Meshes,
    adjacent_faces: bool = True,
) -> torch.Tensor:  # (V, num_vertices) booleano
    """
    :param pix_to_face: tensor pix_to_face de los fragments de MeshRendererWithFragments
    :param mesh: objeto pytorch3d.structures.Meshes
    :param adjacent_faces: si True, tambien marca caras adyacentes como
        visibles (ayuda con triangulos pequenos que el rasterizador puede
        no cubrir completamente)

    :return: mascara booleana (V, num_vertices), V = numero de vistas renderizadas
    """
    num_views = pix_to_face.shape[0]
    faces_packed = mesh.faces_packed()
    num_faces = len(faces_packed)
    num_verts = len(mesh.verts_packed())

    # reshape pix_to_face y manejar valores negativos (sin cara visible)
    visible_faces_per_view = pix_to_face.view(num_views, -1) % num_faces
    valid_mask = visible_faces_per_view >= 0
    visible_faces_per_view = visible_faces_per_view * valid_mask

    if adjacent_faces:
        adjacent_faces_ = find_adjacent_faces(faces_packed)
        visible_faces_per_view = torch.cat(
            [
                visible_faces_per_view,
                adjacent_faces_[visible_faces_per_view].view(num_views, -1),
            ],
            dim=1,
        )

    visible_vertices = faces_packed[visible_faces_per_view.type(torch.long)].view(
        num_views, -1
    )

    visible_vertices_per_view = torch.zeros(
        num_views, num_verts, dtype=torch.bool, device=mesh.device
    )
    # scatter vectorizado en vez de loop para marcar vertices visibles
    visible_vertices_per_view.scatter_(1, visible_vertices, True)

    return visible_vertices_per_view


def rotate_and_interleave_images(
    images: torch.Tensor,  # (V, H, W, 3)
    rotation_indices: list[int],  # 0=0°, 1=90°, 2=180°, 3=270°
) -> torch.Tensor:  # (len(rotation_indices)*V, H, W, 3)
    """
    Aplica las rotaciones indicadas por `rotation_indices` a cada imagen
    y las intercala, de forma que las len(rotation_indices) rotaciones
    de una misma vista queden contiguas en el batch resultante.
    """
    all_rotations = [
        images,  # 0 grados
        torch.rot90(images, k=1, dims=(1, 2)),  # 90 grados
        torch.rot90(images, k=2, dims=(1, 2)),  # 180 grados
        torch.rot90(images, k=3, dims=(1, 2)),  # 270 grados
    ]
    selected_rotations = [all_rotations[idx] for idx in rotation_indices]

    rotations = torch.stack(selected_rotations, dim=0)  # (R, V, H, W, 3)
    rotations = rotations.permute(1, 0, 2, 3, 4)  # (V, R, H, W, 3)
    return rotations.reshape(
        len(rotation_indices) * images.shape[0], *images.shape[1:]
    )  # (R*V, H, W, 3)


def rotate_and_interleave_coordinates(
    pixel_coords: torch.Tensor,  # (V, N, 3)
    image_size: tuple[int, int],  # (H, W)
    rotation_indices: list[int],  # 0=0°, 1=90°, 2=180°, 3=270°
) -> torch.Tensor:  # (len(rotation_indices)*V, N, 3)
    """
    Rota las coordenadas de pixel para que coincidan con las rotaciones
    aplicadas por `rotate_and_interleave_images`, intercaladas de la
    misma forma.
    """
    _, N, _ = pixel_coords.shape
    H, W = image_size

    coords_0 = pixel_coords

    coords_90 = pixel_coords.clone()
    coords_90[:, :, 0] = H - 1 - pixel_coords[:, :, 1]
    coords_90[:, :, 1] = pixel_coords[:, :, 0]

    coords_180 = pixel_coords.clone()
    coords_180[:, :, 0] = W - 1 - pixel_coords[:, :, 0]
    coords_180[:, :, 1] = H - 1 - pixel_coords[:, :, 1]

    coords_270 = pixel_coords.clone()
    coords_270[:, :, 0] = pixel_coords[:, :, 1]
    coords_270[:, :, 1] = W - 1 - pixel_coords[:, :, 0]

    all_rotations = [coords_0, coords_90, coords_180, coords_270]
    selected_rotations = [all_rotations[idx] for idx in rotation_indices]

    stacked = torch.stack(selected_rotations, dim=1)  # (V, R, N, 3)
    return stacked.view(-1, N, 3)  # (R*V, N, 3)


def get_feature_for_pixel_location_optimized_2(
    feature_map: torch.Tensor,  # (V, num_patches, emb_dim)
    pixel_locations: torch.Tensor,  # (V, N, 2) o (V, N, 3)
    image_size: int = 224,
    patch_size: int = 14,
) -> torch.Tensor:  # (V, N, emb_dim)
    """
    Mapea features del feature map a los vertices 3D segun su ubicacion
    de pixel proyectada.
    """
    V, _, emb_dim = feature_map.shape
    _, N, _ = pixel_locations.shape

    pixel_coords = pixel_locations[..., :2]

    num_patches = image_size // patch_size
    patch_coords = pixel_coords / patch_size

    patch_indices_x = torch.clamp(
        patch_coords[..., 0].floor(), 0, num_patches - 1
    ).long()
    patch_indices_y = torch.clamp(
        patch_coords[..., 1].floor(), 0, num_patches - 1
    ).long()

    patch_indices = patch_indices_y * num_patches + patch_indices_x

    batch_indices = torch.arange(V, device=feature_map.device)[:, None].expand(-1, N)

    vertex_features = feature_map[batch_indices, patch_indices]
    return vertex_features


def _ensure_gray_material(mesh: Meshes, color: float = 0.5) -> Meshes:
    if mesh.textures is not None:
        return mesh

    mesh.textures = TexturesVertex(
        verts_features=torch.ones_like(mesh.verts_packed()[None]) * 0.7
    )
    return mesh


def features_backprojection(
    model: DINOWrapper,
    mesh: Meshes,
    views: np.ndarray,
    config: FeatureConfig,
    device: str = "cpu",
) -> torch.Tensor:  # (num_vertices, emb_dim)
    """
    :param model: in: (N, H, W, C), out: (N, num_patches, emb_dim)
    :param mesh: mesh pytorch3d
    :param views: viewpoints ('sample_fibonacci_viewpoints')
    :param config: FeatureConfig (resolution, fov, batch_size, rot_aug)
    :param device: GPU/CPU

    :return: vertex features, averaged over all views
    """
    mesh = mesh.to(device)
    mesh = _normalize_mesh_to_unit_sphere(mesh)
    mesh = _ensure_gray_material(mesh)
    rotation_indices = rotation_maps.get(config.rot_aug, [0])

    R, T = look_at_view_transform(eye=views, device=device)

    raster_settings = RasterizationSettings(
        image_size=(config.resolution, config.resolution),
        faces_per_pixel=5,
        bin_size=0,
        cull_backfaces=True,
    )
    renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardFlatShader(device=device),
    )

    points = mesh.verts_packed()  # (N, 3)
    overall_visibility = torch.zeros(len(points))
    point_values_counts = torch.zeros(len(points), device=device)
    ret_array = None  # se inicializa cuando se conoce emb_dim

    remaining_views = views
    remaining_R, remaining_T = R, T

    while len(remaining_views) > 0:
        batch_views = remaining_views[: config.view_batch_size]
        batch_R = remaining_R[: config.view_batch_size]
        batch_T = remaining_T[: config.view_batch_size]

        remaining_views = remaining_views[config.view_batch_size :]
        remaining_R = remaining_R[config.view_batch_size :]
        remaining_T = remaining_T[config.view_batch_size :]

        okay = False
        while not okay:
            try:
                cameras = FoVPerspectiveCameras(
                    R=batch_R, T=batch_T, fov=config.fov, device=device
                )
                batch_views = batch_views / np.linalg.norm(batch_views, axis=1)[:, None]
                lights = DirectionalLights(direction=batch_views, device=device)

                with torch.no_grad():
                    # images (V, res, res, 4), pix_to_face (V, res, res, 5)
                    images, fragments = renderer(
                        mesh.extend(len(batch_views)), cameras=cameras, lights=lights
                    )
                    images = images[..., :3]  # (V, res, res, 3)

                # (V, N, 3)
                pixel_coords_all_points = cameras.transform_points_screen(
                    points, image_size=(config.resolution, config.resolution)
                ).cpu()
                backprojection_logger.debug(
                    f"points_shape: {points.shape}, batch_r_shape: {batch_R.shape}, batch_T_shape: {batch_T.shape}"
                )
                backprojection_logger.debug(
                    f"pixel_coords_all_points.shape: {pixel_coords_all_points.shape}"
                )
                images = rotate_and_interleave_images(images, rotation_indices)
                pixel_coords_all_points = rotate_and_interleave_coordinates(
                    pixel_coords_all_points,
                    (config.resolution, config.resolution),
                    rotation_indices,
                )

                # (V, N) mascara de visibilidad por vertice y por imagen
                visible_points = check_visible_vertices_optimized(
                    fragments.pix_to_face, mesh
                )
                visible_points = visible_points.repeat_interleave(
                    len(rotation_indices), dim=0
                )
                overall_visibility += visible_points.cpu().sum(dim=0)

                with torch.no_grad():
                    # (V*len(rotation_indices), num_patches, emb_dim)
                    processed_images = model(images)

                # (V*len(rotation_indices), N, emb_dim)
                features_per_view = get_feature_for_pixel_location_optimized_2(
                    processed_images,
                    pixel_coords_all_points,
                    image_size=config.resolution,
                    patch_size=model.patch_size(),
                )

                if ret_array is None:
                    ret_array = torch.zeros(
                        len(points), features_per_view.shape[-1], device=device
                    )

                ret_array += torch.sum(
                    features_per_view * visible_points[..., None], dim=0
                )
                point_values_counts += visible_points.sum(dim=0)

                okay = True

            except AssertionError as e:
                # retry con camara/vistas ligeramente escaladas
                backprojection_logger.warning(f"Retry tras AssertionError: {e}")
                batch_T = batch_T * 1.1
                batch_views = batch_views * 1.1

    if torch.any(overall_visibility == 0):
        backprojection_logger.warning(
            f"{torch.sum(overall_visibility == 0)} vertices no son visibles en ninguna vista"
        )

    ret_array[point_values_counts > 0] /= point_values_counts[point_values_counts > 0][
        :, None
    ]

    backprojection_logger.info(f"RM done | features={ret_array.shape}")
    return ret_array


def _normalize_mesh_to_unit_sphere(mesh: Meshes) -> Meshes:
    verts = mesh.verts_packed()
    center = verts.mean(dim=0)
    mesh = mesh.offset_verts_(-center)
    max_dist = mesh.verts_packed().norm(dim=1).max()
    mesh.scale_verts_(1.0 / max_dist.item())
    return mesh
