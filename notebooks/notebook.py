import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from data_loaders.dataload import load_batch_point_clouds
    from feature_extractor.sampling import sample_fibonacci_views
    from feature_extractor.config import FeatureConfig
    from feature_extractor.rendering import render_point_cloud
    from feature_extractor.backprojection import aggregate_features
    from pytorch3d.structures import Pointclouds
    import plotly.graph_objects as go
    import torch
    import torchvision
    from model_wrappers import DINOWrapper

    return (
        DINOWrapper,
        FeatureConfig,
        aggregate_features,
        go,
        load_batch_point_clouds,
        render_point_cloud,
        sample_fibonacci_views,
        torch,
    )


@app.cell
def _(
    DINOWrapper,
    FeatureConfig,
    aggregate_features,
    load_batch_point_clouds,
    render_point_cloud,
    sample_fibonacci_views,
    torch,
):
    obj_16 = load_batch_point_clouds(1)
    print(obj_16.shape)
    pc = obj_16[3]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    pc = pc.to(device)
    model = DINOWrapper(device=device, small=True, reg=True)
    config = FeatureConfig()
    # Fibonacci sampling views & camera
    R, T = sample_fibonacci_views(pc, config)
    rendered_images, mappings = render_point_cloud(pc, R, T, config)
    _render = rendered_images
    model_outputs = model(rendered_images)
    features = aggregate_features(model_outputs, mappings, pc, FeatureConfig())
    return R, T, features, mappings, model_outputs, obj_16, pc, rendered_images


@app.cell
def _(obj_16):
    pc_ = obj_16[6]
    pc_
    return (pc_,)


@app.cell
def _():
    return


@app.cell
def _(go, pc_):

    x, y, z = pc_[:, 0], pc_[:, 1], pc_[:, 2]

    # 4. Crear la gráfica con Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=z,                # Color basado en la coordenada Z
            colorscale='Viridis',   # Escala de colores
            opacity=0.8
        )
    )])

    fig.update_layout(
        title="Visualización de Tensor 3D",
        scene=dict(
            xaxis_title='Dimensión 1',
            yaxis_title='Dimensión 2',
            zaxis_title='Dimensión 3'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()
    return


@app.cell
def _(R, T, features, mappings, model_outputs, pc, rendered_images, torch):
    from pathlib import Path

    fixtures_dir = Path("test/fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    torch.save(pc, fixtures_dir / "point_cloud.pt")
    torch.save(R, fixtures_dir / "R.pt")
    torch.save(T, fixtures_dir / "T.pt")
    torch.save(rendered_images, fixtures_dir / "rendered_images.pt")
    torch.save(model_outputs, fixtures_dir / "model_outputs.pt")
    torch.save(mappings, fixtures_dir / "mappings.pt")
    torch.save(features, fixtures_dir / "features.pt")
    return


@app.cell
def _(torch):
    features_pipe = "./test/fixtures/features.pt"
    features_original = "./test/fixtures/features_original.pt"
    original = torch.load(features_original, map_location="cpu")
    refactored = torch.load(features_pipe, map_location="cpu")

    diff = (original - refactored).abs()
    print("max:", diff.max().item())
    print("mean:", diff.mean().item())
    print("median:", diff.median().item())
    print("% de elementos con diff > 0.01:", (diff > 0.01).float().mean().item() * 100)
    diff_ = (original - refactored).abs().sum(dim=-1)  # diferencia por punto, no por feature individual
    worst_points = diff_.topk(10).indices
    print(worst_points)

    # verificar si esos puntos son justamente los que fueron interpolados (ocluidos)
    # comparando contra mappings o el conteo de visibilidad si lo guardaste
    return


@app.cell
def _(torch):
    from pathlib import Path
    from dotenv import load_dotenv
    import os
    import numpy as np
    import pandas as pd
    load_dotenv()
    objs_path = Path(os.environ["DATA_SYM_PLANE_PROCESSED"])

    statistics = []
    a = True
    for obj in objs_path.iterdir():
        if obj.suffix == ".npy":
            point_cloud = torch.from_numpy(np.load(obj))
            if a:
                print(point_cloud.shape)
                a = False
            mean = point_cloud.mean(dim=0)              # (3,) - centro de masa
            radius = point_cloud.norm(dim=-1).max()      # escalar - distancia maxima al origen
            bbox_min = point_cloud.min(dim=0).values     # (3,) - extremos del bounding box
            bbox_max = point_cloud.max(dim=0).values
            statistics.append({"name": obj.stem, "mean_x": mean[0].item(), "mean_y": mean[1].item(), "mean_z": mean[2].item(), "radius": radius.item()})
    df = pd.DataFrame(statistics)
    df[["mean_x", "mean_y", "mean_z", "radius"]].describe()
    return


if __name__ == "__main__":
    app.run()
