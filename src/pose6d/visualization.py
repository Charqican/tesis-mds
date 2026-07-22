from dataclasses import dataclass
from pathlib import Path

import open3d as o3d
import numpy as np

from .registrate import RegistrationResult


# ----------------------------------------------------------------------------- PLY writer


def write_ply(
    path: str | Path, pts: np.ndarray, colors: np.ndarray | None = None
) -> None:
    """Escribe un PLY ASCII point cloud. colors: (N, 3) uint8 o None."""
    pts = np.asarray(pts, dtype=np.float32)
    n = len(pts)
    has_c = colors is not None
    if has_c:
        colors = np.asarray(colors, dtype=np.uint8)

    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_c:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(n):
            if has_c:
                f.write(
                    f"{pts[i, 0]} {pts[i, 1]} {pts[i, 2]} "
                    f"{colors[i, 0]} {colors[i, 1]} {colors[i, 2]}\n"
                )
            else:
                f.write(f"{pts[i, 0]} {pts[i, 1]} {pts[i, 2]}\n")


def save_registration_plys(
    result: RegistrationResult,
    out_dir: str | Path,
    prefix: str = "",
) -> dict[str, Path]:
    """
    Guarda 4 archivos PLY para visualización externa.
    Retorna dict con paths creados.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grey = np.tile([160, 160, 160], (len(result.scene_pts), 1))
    red = np.tile([220, 40, 40], (len(result.posed_mesh_pts), 1))

    paths = {}

    # Scene
    p = out_dir / f"{prefix}scene.ply"
    write_ply(p, result.scene_pts, grey)
    paths["scene"] = p

    # Posed mesh
    p = out_dir / f"{prefix}object_mesh_posed.ply"
    write_ply(p, result.posed_mesh_pts, red)
    paths["posed_mesh"] = p

    # Object points (if available)
    if result.object_pts is not None and len(result.object_pts) > 0:
        green = np.tile([40, 200, 40], (len(result.object_pts), 1))
        p = out_dir / f"{prefix}object_scene_pts.ply"
        write_ply(p, result.object_pts, green)
        paths["object_pts"] = p

        # Combined
        combo_pts = [result.scene_pts, result.posed_mesh_pts, result.object_pts]
        combo_col = [grey, red, green]
    else:
        combo_pts = [result.scene_pts, result.posed_mesh_pts]
        combo_col = [grey, red]

    p = out_dir / f"{prefix}combined.ply"
    write_ply(p, np.concatenate(combo_pts), np.concatenate(combo_col))
    paths["combined"] = p

    return paths


# ----------------------------------------------------------------------------- Open3D viewer


def visualize_open3d(result: RegistrationResult, point_size: float = 2.0) -> None:
    """Visualización interactiva con Open3D (ventana OpenGL)."""

    def make_pc(pts, color):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(pts)
        pc.paint_uniform_color(color)
        return pc

    geoms = [
        make_pc(result.scene_pts, [0.6, 0.6, 0.6]),
        make_pc(result.posed_mesh_pts, [0.85, 0.15, 0.15]),
    ]

    if result.object_pts is not None and len(result.object_pts) > 0:
        geoms.append(make_pc(result.object_pts, [0.15, 0.8, 0.15]))

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50.0)
    o3d.visualization.draw_geometries(geoms + [axis])


# ----------------------------------------------------------------------------- Plotly viewer


def visualize_plotly(result: RegistrationResult, title: str = "Registration") -> None:
    """Visualización en navegador con Plotly (funciona en cualquier lado)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Scene
    fig.add_trace(
        go.Scatter3d(
            x=result.scene_pts[:, 0],
            y=result.scene_pts[:, 1],
            z=result.scene_pts[:, 2],
            mode="markers",
            marker=dict(size=1, color="grey", opacity=0.3),
            name="Scene",
        )
    )

    # Posed mesh
    fig.add_trace(
        go.Scatter3d(
            x=result.posed_mesh_pts[:, 0],
            y=result.posed_mesh_pts[:, 1],
            z=result.posed_mesh_pts[:, 2],
            mode="markers",
            marker=dict(size=2, color="red", opacity=0.8),
            name="Posed mesh",
        )
    )

    # Object points
    if result.object_pts is not None and len(result.object_pts) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=result.object_pts[:, 0],
                y=result.object_pts[:, 1],
                z=result.object_pts[:, 2],
                mode="markers",
                marker=dict(size=2, color="green", opacity=0.8),
                name="Object visible pts",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.show()


# ----------------------------------------------------------------------------- Unified entry point


def visualize(
    result: RegistrationResult,
    method: str = "auto",
    out_dir: str | Path | None = None,
    title: str = "Registration",
) -> None:
    """
    Visualiza el registro. Intenta Open3D primero, cae a Plotly.

    Args:
        method: "auto" (intentar Open3D, caer a Plotly), "open3d", "plotly", "ply"
        out_dir: si method="ply", directorio de salida
        title: título para Plotly
    """
    if method == "ply":
        if out_dir is None:
            raise ValueError("out_dir required for method='ply'")
        paths = save_registration_plys(result, out_dir)
        print(f"PLYs saved: {paths}")
        return

    if method == "open3d":
        visualize_open3d(result)
        return

    if method == "plotly":
        visualize_plotly(result, title)
        return

    # auto: intentar Open3D, caer a Plotly
    try:
        visualize_open3d(result)
    except Exception as e:
        print(f"Open3D failed ({e}), falling back to Plotly...")
        visualize_plotly(result, title)
