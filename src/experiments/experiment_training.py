import copy
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader

from pose6d.dataset import SplitData
from logger import TimedLogger, training_logger as log


@dataclass
class TrainResult:
    model: torch.nn.Module  # best model (by val loss), in eval mode
    best_epoch: int
    best_val_loss: float
    n_epochs: int


def instance_losses(pred: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(N, K) -> (N,) mean squared error per instance, in normalized space."""
    return ((pred - targets) ** 2).mean(dim=1)


@torch.no_grad()
def predict(model: torch.nn.Module, split: SplitData, device, chunk: int = 256):
    """Normalized predictions (N, K) of a whole split, on cpu."""
    model.eval()
    preds = [
        model(split.inputs[i : i + chunk].to(device)).cpu()
        for i in range(0, len(split), chunk)
    ]
    return torch.cat(preds)


def train(
    model: torch.nn.Module,
    train_split: SplitData,
    val_split: SplitData,
    batch_size: int,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs: dict | None = None,
    scheduler_cls=None,
    scheduler_kwargs: dict | None = None,
    max_epochs: int = 5000,
    min_epochs: int = 0,
    patience: int | None = None,
    min_delta: float = 0.0,
    eval_every: int = 10,
    on_eval=None,
    device=None,
) -> TrainResult:
    """
    Early stopping in epochs: stops once val loss has not improved (by a relative
    min_delta) for `patience` epochs, counted from max(best epoch, min_epochs).
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader = DataLoader(train_split, batch_size=batch_size, shuffle=True)
    optimizer = optimizer_cls(model.parameters(), **(optimizer_kwargs or {}))
    scheduler = (
        scheduler_cls(optimizer, **(scheduler_kwargs or {})) if scheduler_cls else None
    )
    progress = TimedLogger(log, every_s=60)

    best_val_loss, best_epoch, best_state = float("inf"), 0, None
    val_loss = float("nan")
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = instance_losses(model(inputs), targets).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_loss = epoch_loss / len(loader)
        if scheduler:
            scheduler.step()

        if epoch % eval_every == 0:
            pred = predict(model, val_split, device)
            val_loss = instance_losses(pred, val_split.targets).mean().item()
            if on_eval:
                on_eval({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            if val_loss < best_val_loss * (1 - min_delta):
                best_val_loss, best_epoch = val_loss, epoch
                best_state = copy.deepcopy(model.state_dict())

        progress(
            f"epoch {epoch}/{max_epochs} train {train_loss:.4f} val {val_loss:.4f} "
            f"best {best_val_loss:.4f} @ {best_epoch} "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )
        if (
            patience is not None
            and epoch >= min_epochs
            and epoch - max(best_epoch, min_epochs) >= patience
        ):
            break

    log.info(f"Stopped at epoch {epoch}, best val {best_val_loss:.4f} @ {best_epoch}")
    model.load_state_dict(best_state)
    return TrainResult(model.eval(), best_epoch, best_val_loss, epoch + 1)
