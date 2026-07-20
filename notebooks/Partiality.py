import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from data_loaders.SimplePointCloudLoader import PointCloudLoader
    from data_loaders.config import DataLoaderConfig
    from degradation.degradations import RandomViewpointDegradation, EquatorCurveDegradation, TransverseMeridianCurveDegradation, MeridianCurveDegradation
    from ProjPaths import ProjPath
    from dotenv import load_dotenv
    from pathlib import Path
    import plotly.graph_objects as go
    import plotly.express as px
    import torch

    return (
        DataLoaderConfig,
        EquatorCurveDegradation,
        Path,
        PointCloudLoader,
        ProjPath,
        go,
        mo,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Parcialidad de nubes de puntos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Config
    """)
    return


@app.cell
def _(
    DataLoaderConfig,
    EquatorCurveDegradation,
    Path,
    PointCloudLoader,
    ProjPath,
):
    paths = ProjPath()
    pc_path = Path(paths.gt_plane_symm)
    gt_path = Path(paths.gt_plane_symm_ori)
    settings = DataLoaderConfig(batch_size=1)
    dataloader = PointCloudLoader(settings, pc_path, gt_path=gt_path)
    strategy = EquatorCurveDegradation()
    return dataloader, strategy


@app.cell
def _(dataloader):
    next(dataloader.batches(gt = True))
    next(dataloader.batches(gt = True))
    names, symmetry_planes, pc = next(dataloader.batches(gt = True))
    print(symmetry_planes[0][0])
    return pc, symmetry_planes


@app.cell
def _(pc, strategy, symmetry_planes):
    pc_deg, mask = strategy(pc[0], symmetry_planes[0][0], 0, .15 )
    print(pc_deg.shape, mask.shape)
    return (pc_deg,)


@app.cell
def _(go, torch):
    def plot_occlusion(
        points: torch.Tensor,
        plane: tuple[torch.Tensor, torch.Tensor],
        viewpoint: torch.Tensor,
        k: float,
    ) -> go.Figure:
        """
        :param points: nube de puntos degradada, (3, N) o (N, 3)
        :param plane: (normal, point) del plano de simetria, cada uno (3,)
        :param viewpoint: viewpoint usado para la oclusion, (3,)
        :param k: fraccion removida, para el titulo
        """
        # Asegurar que points sea (N, 3) para plotly
        if points.dim() == 2:
            if points.shape[0] == 3 and points.shape[1] != 3:
                # (3, N) -> (N, 3)
                points_np = points.detach().cpu().numpy().T
            else:
                # (N, 3)
                points_np = points.detach().cpu().numpy()
        else:
            points_np = points.detach().cpu().numpy()
            if points_np.ndim != 2 or points_np.shape[1] != 3:
                raise ValueError(f"points debe ser (N, 3) o (3, N), got shape {points_np.shape}")

        # Aplanar explícitamente para evitar problemas de broadcasting
        x = points_np[:, 0].flatten()
        y = points_np[:, 1].flatten()
        z = points_np[:, 2].flatten()

        normal, point = plane
        normal_np = normal.detach().cpu().numpy().flatten()
        point_np = point.detach().cpu().numpy().flatten()
        viewpoint_np = viewpoint.detach().cpu().numpy().flatten()

        # --- Construir base ortonormal del plano ---
        reference = torch.tensor([1.0, 0.0, 0.0], dtype=normal.dtype, device=normal.device)
        if torch.abs(torch.dot(normal / normal.norm(), reference)) > 0.9:
            reference = torch.tensor([0.0, 1.0, 0.0], dtype=normal.dtype, device=normal.device)
        n_unit = normal / normal.norm()
        u = reference - torch.dot(reference, n_unit) * n_unit
        u = u / u.norm()
        v = torch.cross(n_unit, u, dim=-1)
        u_np = u.detach().cpu().numpy().flatten()
        v_np = v.detach().cpu().numpy().flatten()

        half = 1.0
        corners = [
            point_np + half * u_np + half * v_np,
            point_np + half * u_np - half * v_np,
            point_np - half * u_np - half * v_np,
            point_np - half * u_np + half * v_np,
        ]

        fig = go.Figure()

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(size=2, color="royalblue"),
                name="Puntos",
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[viewpoint_np[0]],
                y=[viewpoint_np[1]],
                z=[viewpoint_np[2]],
                mode="markers",
                marker=dict(size=6, color="crimson", symbol="diamond"),
                name="Viewpoint",
            )
        )

        fig.add_trace(
            go.Mesh3d(
                x=[c[0] for c in corners],
                y=[c[1] for c in corners],
                z=[c[2] for c in corners],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="mediumseagreen",
                opacity=0.25,
                name="Plano de simetria",
            )
        )

        fig.update_layout(
            title=f"Oclusion (k={k * 100:.1f}%)",
            scene=dict(aspectmode="manual"),
            margin=dict(l=0, r=0, t=40, b=0),
        )

        return fig

    return (plot_occlusion,)


@app.cell
def _(pc_deg, plot_occlusion, strategy, symmetry_planes):
    plot_occlusion(
        pc_deg,
        symmetry_planes[0][0],
        strategy.view_point,
        .3,
    ) 
    return


if __name__ == "__main__":
    app.run()
