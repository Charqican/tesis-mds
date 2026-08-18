from pathlib import Path
import torch

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pose6d.config import LMOConfig
from pose6d.data_loader import LMOLoader
from pose6d.training import SymmetryPointDataset
from pose6d.model import SymmetryFieldMLP

CACHE_ROOT = Path("/mnt/data/dev/dataset/tesis/6dpose/cache")
POINTS_PT_DIR = CACHE_ROOT / "points_pT"
FEATURES_DGEDI_DIR = CACHE_ROOT / "features_dgedi_pT"

N_EPOCHS = 1600
LR = 1e-3
LOG_EVERY = 25


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path("/mnt/data/dev/dataset/tesis/BOP/lmo/lmo")
    scene_id = 2

    config = LMOConfig.from_root(root)
    loader = LMOLoader(config)

    dataset = SymmetryPointDataset(config, loader, POINTS_PT_DIR, FEATURES_DGEDI_DIR)
    print(f"Puntos totales: {len(dataset)}")
    print(f"target_mean={dataset.target_mean:.4f}  target_std={dataset.target_std:.4f}")
    HELD_OUT_UIDS = {
        "scene000002_img000326_obj000011_inst05",
        "scene000002_img000224_obj000010_inst05",
    }
    # select mask to substract from training data
    held_out_mask = torch.tensor(
        [dataset.uid_list[i.item()] in HELD_OUT_UIDS for i in dataset.instance_idx]
    )
    # obtain train mask
    train_mask = ~held_out_mask

    features = dataset.features[train_mask].to(device)
    targets = dataset.targets[train_mask].to(device)  # ya normalizado

    model = SymmetryFieldMLP(in_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # subido de 1e-3
    loss_fn = torch.nn.MSELoss()

    first_loss, last_loss = None, None
    for epoch in range(N_EPOCHS):
        optimizer.zero_grad()
        pred = model(features)
        loss = loss_fn(pred, targets)
        loss.backward()
        optimizer.step()

        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()

        if epoch % LOG_EVERY == 0 or epoch == N_EPOCHS - 1:
            print(f"epoch {epoch:4d}  loss(norm)={loss.item():.6f}")

    print(f"\nloss inicial(norm)={first_loss:.6f}  loss final(norm)={last_loss:.6f}")

    # visualization of held-out + training to compare
    with torch.no_grad():
        for uid in HELD_OUT_UIDS:
            pts, feats, target_raw = dataset.get_instance(uid)
            pred_raw = dataset.denormalize(model(feats.to(device)).cpu())
            plot_target_vs_pred(pts, target_raw, pred_raw, title=f"held-out: {uid}")

        train_uid = dataset.uid_list[0]
        pts, feats, target_raw = dataset.get_instance(train_uid)
        pred_raw = dataset.denormalize(model(feats.to(device)).cpu())
        plot_target_vs_pred(pts, target_raw, pred_raw, title=f"Training: {train_uid}")
        # pred_mm = dataset.denormalize(model(features).cpu())
        # rmse_mm = torch.sqrt(torch.mean((pred_mm - dataset.targets_raw) ** 2))
        # print(f"RMSE final en mm: {rmse_mm:.4f}")


def plot_target_vs_pred(
    points: torch.Tensor, target: torch.Tensor, pred: torch.Tensor, title: str
):
    points_np = points.numpy()
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Target (GT)", "Predicción"),
    )
    common = dict(
        x=points_np[:, 0],
        y=points_np[:, 1],
        z=points_np[:, 2],
        mode="markers",
    )
    fig.add_trace(
        go.Scatter3d(
            **common, marker=dict(size=2, color=target.numpy(), colorscale="RdBu")
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            **common, marker=dict(size=2, color=pred.numpy(), colorscale="RdBu")
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=title, height=600)
    fig.show()


if __name__ == "__main__":
    main()
