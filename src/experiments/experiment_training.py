import copy
from experiments.experiment_setup import TrainData
from torch.utils.data import DataLoader
from pose6d.dataset import SymmetryFieldInstanceDataset
import torch


def training_function(
    training_data: TrainData,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
    scheduler_cls=None,
    scheduler_kwargs=None,
    n_epochs=5000,
    log_every=None,
    on_epoch=None,
    patience: int | None = None,
    min_delta: float = 0.002,
    restore_best: bool = True,
    device=None,
    min_epochs: int = 0,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset: SymmetryFieldInstanceDataset = training_data.dataset
    if not dataset._loaded:
        dataset.load()

    train_loader = DataLoader(
        dataset.get_split("train"), batch_size=training_data.batch_size, shuffle=True
    )
    val_idx = dataset.indices_for("val")
    if not val_idx:
        raise ValueError(
            "no val split assigned; pass val_uids in assign_splits_explicit"
        )
    val_input = dataset.input[val_idx].to(device)
    val_targets = dataset.targets[val_idx].to(device)

    model = training_data.model.to(device)
    optimizer = optimizer_cls(model.parameters(), **(optimizer_kwargs or {}))
    scheduler = (
        scheduler_cls(optimizer, **(scheduler_kwargs or {})) if scheduler_cls else None
    )
    log_every = log_every or max(n_epochs // 250, 1)

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state = None

    loss_history, loss_val_history = [], []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for inp, targets in train_loader:
            inp, targets = inp.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(inp.reshape(-1, training_data.input_dim)).reshape(
                targets.shape
            )
            loss = ((pred - targets) ** 2).mean(dim=1).mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        if scheduler:
            scheduler.step()

        if epoch % log_every == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(
                    val_input.reshape(-1, training_data.input_dim)
                ).reshape(val_targets.shape)
                mean_val_loss = (
                    ((val_pred - val_targets) ** 2).mean(dim=1).mean().item()
                )
            model.train()

            mean_train_loss = epoch_loss / n_batches
            loss_history.append(mean_train_loss)
            loss_val_history.append(mean_val_loss)
            if on_epoch:
                on_epoch(
                    {"train_loss": mean_train_loss, "val_loss": mean_val_loss},
                    step=epoch,
                )

            if mean_val_loss < best_val_loss * (1 - min_delta):
                best_val_loss = mean_val_loss
                epochs_without_improvement = 0
                if restore_best:
                    best_state = copy.deepcopy(model.state_dict())
            elif epoch >= min_epochs:
                epochs_without_improvement += 1

            if (
                patience is not None
                and epoch >= min_epochs
                and epochs_without_improvement >= patience
            ):
                break
    if restore_best and best_state is not None:
        model.load_state_dict(best_state)

    return model, loss_history, loss_val_history
