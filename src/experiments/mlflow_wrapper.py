import mlflow
import mlflow.pytorch as mlflowpy
from mlflow.models import infer_signature
from logger import notebook_logger as log


def run_experiment(config: dict, setup_func, train_eng):
    """
    # Params:

    config: dict
        dictionary with information about the experiment.
        - 'experiment_name'
        - 'setup_conf'
        - 'train_conf'
    """
    mlflow.set_experiment(config["experiment_name"])
    with mlflow.start_run(run_name=config.get("run_name")):
        mlflow.log_params(flatten(config))
        train_data = setup_func(**config["setup_conf"])
        model, loss_hist, test_hist = train_eng(
            train_data,
            on_epoch=lambda m, step: mlflow.log_metrics(m, step=step),
            **config["train_conf"],
        )
        x_feat, _ = train_data.dataset[0]
        # log.info(x_feat)
        signature = infer_signature(
            x_feat.to("cpu").numpy()
        )  # o con predictions también
        mlflowpy.log_model(
            train_data.model.eval().to("cpu"),
            "model",
            input_example=x_feat.to("cpu"),
            signature=signature,
        )

        mlflow.log_metric("final_test_loss", test_hist[-1])
    return loss_hist, test_hist, train_data.model, train_data.dataset


### UTILS:


# logger function for formatting mlflow
def flatten(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
