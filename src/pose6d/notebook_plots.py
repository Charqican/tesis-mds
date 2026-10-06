import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import trimesh
from sklearn.decomposition import PCA

from pose6d.loader import BOPLoader
from pose6d.geometry_utils import backproject_depth, transform_points
from pose6d.preprocessing import extract_instances_pcs


def _posed_mesh(loader: BOPLoader, scene_id: int, img_id: int, inst_idx: int):
    """(obj_id, posed vertices, faces) of one instance of a frame."""
    instance = loader.load_instances(scene_id, img_id)[inst_idx]
    mesh = trimesh.load(loader.cfg.paths.model_path(instance.obj_id))
    return (
        instance.obj_id,
        transform_points(mesh.vertices, instance.R, instance.t),
        mesh.faces,
    )


def _mesh_trace(vertices: np.ndarray, faces: np.ndarray, **kwargs) -> go.Mesh3d:
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=-vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        **kwargs,
    )


def plot_frame_meshes_and_sensor(
    loader: BOPLoader, scene_id: int, img_id: int
) -> go.Figure:
    config = loader.cfg
    K, depth_scale = loader.load_camera(scene_id, img_id)
    depth_image = loader.load_depth(scene_id, img_id)
    frame_point_cloud = backproject_depth(
        depth_image, K, depth_scale, stride=config.depth_stride
    )

    traces = [
        go.Scatter3d(
            x=frame_point_cloud[:, 0],
            y=frame_point_cloud[:, 1],
            z=-frame_point_cloud[:, 2],
            mode="markers",
            marker=dict(size=1, color="gray", opacity=0.3),
            name="sensor",
        )
    ]

    for inst_idx in range(len(loader.load_instances(scene_id, img_id))):
        obj_id, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)
        traces.append(
            _mesh_trace(
                vertices,
                faces,
                opacity=1,
                name=f"obj_{obj_id} (inst {inst_idx})",
                showlegend=True,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Scene {scene_id} / Frame {img_id}",
        height=700,
        scene=dict(aspectmode="data"),
    )
    return fig


def plot_mesh_instance_visible(
    loader: BOPLoader, scene_id: int, img_id: int, inst_idx: int
) -> go.Figure:
    obj_id, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)
    traces = [
        _mesh_trace(
            vertices,
            faces,
            opacity=0.7,
            name=f"obj : {obj_id} (instance {inst_idx})",
            showlegend=True,
        )
    ]

    pTr_iterator = extract_instances_pcs(loader, scene_id, img_id, [obj_id])
    _, visib_posed_pcs = next(pTr_iterator)

    traces.append(
        go.Scatter3d(
            x=visib_posed_pcs[:, 0],
            y=visib_posed_pcs[:, 1],
            z=-visib_posed_pcs[:, 2],
            mode="markers",
            marker=dict(size=2, color="yellow", opacity=0.5),
            name=f"Visible (instance {inst_idx})",
            showlegend=True,
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Scene {scene_id} | Frame {img_id} | object {obj_id} | instance {inst_idx}",
        height=700,
        scene=dict(aspectmode="data"),
    )
    return fig


# general purpose: symmetry field, error field, or any other per-point scalar
def plot_mesh_with_scalar_field(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    inst_idx: int,
    points: np.ndarray,
    values: np.ndarray,
    colorscale: str = "Viridis",
    colorbar_title: str = "value",
) -> go.Figure:
    obj_id, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)
    traces = [
        _mesh_trace(
            vertices,
            faces,
            opacity=0.7,
            name=f"obj : {obj_id} (instance {inst_idx})",
            showlegend=True,
            color="orange",
        ),
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=-points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=values,
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(title=colorbar_title),
                opacity=0.8,
            ),
            name=f"Visible (instance {inst_idx})",
            showlegend=True,
        ),
    ]

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Scene {scene_id} | Frame {img_id} | object {obj_id} | instance {inst_idx}",
        height=700,
        scene=dict(aspectmode="data"),
    )
    return fig


def features_to_rgb(
    features: np.ndarray,
    reference: np.ndarray | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    reference = (
        features if reference is None else np.asarray(reference, dtype=np.float64)
    )

    pca = PCA(n_components=3).fit(reference)
    signs = np.sign(
        pca.components_[np.arange(3), np.abs(pca.components_).argmax(axis=1)]
    )
    signs[signs == 0] = 1.0
    pca.components_ *= signs[:, None]

    low, high = np.percentile(pca.transform(reference), percentiles, axis=0)
    span = np.where(high - low > 1e-12, high - low, 1.0)

    scaled = (pca.transform(features) - low) / span
    inside = np.all((scaled >= 0.0) & (scaled <= 1.0), axis=1)
    rgb = np.clip(scaled, 0.0, 1.0) * 255.0
    colors = np.array([f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in rgb])

    return colors, inside, pca.explained_variance_ratio_


def plot_mesh_instance_with_dgedi_features(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    inst_idx: int,
    points: np.ndarray,
    features: np.ndarray,
    reference: np.ndarray | None = None,
    show_mesh: bool = True,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> go.Figure:
    obj_id, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)
    colors, inside, evr = features_to_rgb(features, reference, percentiles=percentiles)
    outside = ~inside

    traces = []
    if show_mesh:
        traces.append(
            _mesh_trace(
                vertices,
                faces,
                opacity=0.3,
                name=f"obj : {obj_id} (instance {inst_idx})",
                showlegend=True,
            )
        )
    traces.append(
        go.Scatter3d(
            x=points[inside, 0],
            y=points[inside, 1],
            z=-points[inside, 2],
            mode="markers",
            marker=dict(size=3, color=colors[inside].tolist(), opacity=0.9),
            name=f"dGeDi PCA ({int(inside.sum())} pts)",
            showlegend=True,
        )
    )
    if outside.any():
        traces.append(
            go.Scatter3d(
                x=points[outside, 0],
                y=points[outside, 1],
                z=-points[outside, 2],
                mode="markers",
                marker=dict(size=3, color="rgb(130,130,130)", symbol="x", opacity=0.9),
                name=f"fuera de rango ({int(outside.sum())} pts)",
                showlegend=True,
            )
        )

    evr_text = " | ".join(f"PC{i + 1}: {v:.1%}" for i, v in enumerate(evr))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(
            f"Scene {scene_id} | Frame {img_id} | object {obj_id} | instance {inst_idx}"
            f"<br><sup>RGB = PC1, PC2, PC3 &nbsp;&nbsp; {evr_text}</sup>"
        ),
        height=700,
        scene=dict(aspectmode="data"),
    )
    return fig


def plot_pcd_outlier_mask(
    pcd: np.ndarray, outlier_mask: np.ndarray, uid: str
) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=pcd[~outlier_mask, 0],
                y=pcd[~outlier_mask, 1],
                z=-pcd[~outlier_mask, 2],
                mode="markers",
                marker=dict(size=2, color="steelblue"),
                name="inliers",
            ),
            go.Scatter3d(
                x=pcd[outlier_mask, 0],
                y=pcd[outlier_mask, 1],
                z=-pcd[outlier_mask, 2],
                mode="markers",
                marker=dict(size=4, color="red"),
                name="Detected Outliers",
            ),
        ]
    )
    fig.update_layout(
        title=f"{uid} — statistical outlier removal",
        height=700,
        scene=dict(aspectmode="data"),
    )
    return fig


def mask_error_outliers(error: np.ndarray, std_ratio: float = 2.0) -> np.ndarray:
    mean = error.mean()
    std = error.std()
    if std < 1e-12:
        return np.ones_like(error, dtype=bool)
    return np.abs(error - mean) <= std_ratio * std


def plot_gt_vs_pred_comparison(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    inst_idx: int,
    points: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    points = np.asarray(points)
    target = np.asarray(target)
    pred = np.asarray(pred)

    cmin = float(min(target.min(), pred.min()))
    cmax = float(max(target.max(), pred.max()))

    _, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)

    def _mesh():
        return _mesh_trace(
            vertices, faces, opacity=0.3, color="orange", showlegend=False
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("target", "pred"),
    )
    fig.add_trace(_mesh(), row=1, col=1)
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=-points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=target,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=True,
                colorbar=dict(title="value"),
            ),
            name="target",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(_mesh(), row=1, col=2)
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=-points[:, 2],
            mode="markers",
            marker=dict(
                size=3,
                color=pred,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
            ),
            name="pred",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        title=title,
        height=600,
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )
    return fig


# same idea, but points whose error is a statistical outlier are grayed out instead of colored
def plot_gt_vs_pred_comparison_robust(
    loader: BOPLoader,
    scene_id: int,
    img_id: int,
    inst_idx: int,
    points: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    std_ratio: float = 2.0,
    title: str = "",
    colorscale: str = "Viridis",
) -> go.Figure:
    points = np.asarray(points)
    target = np.asarray(target)
    pred = np.asarray(pred)

    error = pred - target
    inlier = mask_error_outliers(error, std_ratio)
    outlier = ~inlier

    cmin = float(min(target[inlier].min(), pred[inlier].min()))
    cmax = float(max(target[inlier].max(), pred[inlier].max()))

    _, vertices, faces = _posed_mesh(loader, scene_id, img_id, inst_idx)

    def _mesh():
        return _mesh_trace(
            vertices, faces, opacity=0.3, color="orange", showlegend=False
        )

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("target", "pred"),
    )
    fig.add_trace(_mesh(), row=1, col=1)
    fig.add_trace(_mesh(), row=1, col=2)

    def _add(values, name, col, showscale):
        fig.add_trace(
            go.Scatter3d(
                x=points[inlier, 0],
                y=points[inlier, 1],
                z=-points[inlier, 2],
                mode="markers",
                marker=dict(
                    size=3,
                    color=values[inlier],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    showscale=showscale,
                    colorbar=dict(title="value") if showscale else None,
                ),
                name=f"{name} (inlier)",
            ),
            row=1,
            col=col,
        )
        if outlier.any():
            fig.add_trace(
                go.Scatter3d(
                    x=points[outlier, 0],
                    y=points[outlier, 1],
                    z=-points[outlier, 2],
                    mode="markers",
                    marker=dict(size=4, color="rgb(130,130,130)", symbol="x"),
                    name=f"{name} (outlier)",
                ),
                row=1,
                col=col,
            )

    _add(target, "target", 1, showscale=True)
    _add(pred, "pred", 2, showscale=False)

    fig.update_layout(
        title=title,
        height=600,
        scene=dict(aspectmode="data"),
        scene2=dict(aspectmode="data"),
    )
    return fig
