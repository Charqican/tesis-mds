from experiments.experiment_setup import TrainData
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
    device=None,
):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_features = training_data.test_features.to(device)
    test_targets = training_data.test_targets.to(device)
    model = training_data.model.to(device)
    optimizer = optimizer_cls(model.parameters(), **(optimizer_kwargs or {}))
    scheduler = (
        scheduler_cls(optimizer, **(scheduler_kwargs or {})) if scheduler_cls else None
    )
    log_every = log_every or max(n_epochs // 250, 1)

    loss_history, loss_test_history = [], []
    for epoch in range(n_epochs):
        epoch_loss, n_batches = 0.0, 0
        for features, targets in training_data.train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            pred = model(features.reshape(-1, training_data.feature_dim)).reshape(
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
                    test_features.reshape(-1, training_data.feature_dim)
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

    # model is inplace (as to(device) make it so), the model is returned for clarity and tracking
    return model, loss_history, loss_test_history
