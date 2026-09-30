import torch
from torch.utils.data import DataLoader
from experiments.experiment_setup import TrainData
from pose6d.dataset import SymmetryFieldInstanceDataset


def training_function(
    training_data: TrainData,
    optimizer_cls=torch.optim.Adam,
    optimizer_kwargs=None,
    scheduler_cls=None,
    scheduler_kwargs=None,
    n_epochs=5000,
    log_every=None,
    on_epoch=None,
    device=None,
):

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset: SymmetryFieldInstanceDataset = training_data.dataset
    if not dataset._loaded:
        dataset.load()

    train_loader = DataLoader(
        dataset.get_split("train"), batch_size=training_data.batch_size, shuffle=True
    )
    test_idx = dataset.indices_for("test")
    test_input = dataset.input[test_idx].to(device)
    test_targets = dataset.targets[test_idx].to(device)

    model = training_data.model.to(device)
    optimizer = optimizer_cls(model.parameters(), **(optimizer_kwargs or {}))
    scheduler = (
        scheduler_cls(optimizer, **(scheduler_kwargs or {})) if scheduler_cls else None
    )
    log_every = log_every or max(n_epochs // 250, 1)

    loss_history, loss_test_history = [], []
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
                test_pred = model(
                    test_input.reshape(-1, training_data.input_dim)
                ).reshape(test_targets.shape)
                mean_test_loss = (
                    ((test_pred - test_targets) ** 2).mean(dim=1).mean().item()
                )
            model.train()
            loss_history.append(epoch_loss / n_batches)
            loss_test_history.append(mean_test_loss)
            if on_epoch:
                on_epoch(
                    {"loss": loss_history[-1], "test_loss": mean_test_loss}, step=epoch
                )

    return model, loss_history, loss_test_history
