import argparse
import sys
from dotenv import load_dotenv
import questionary
from rich.console import Console
from rich.table import Table

from experiments.registry import REGISTRY, Experiment
from experiments.mlflow_wrapper import run_experiment
from pose6d.layout import DataLayout

load_dotenv()  # this reads the .env mlflow variable before importing mlflow.
import mlflow


console = Console()
"""
Rich CLI to list, run and reset the experiments of experiments/registry.py.
A config counts as done when its run finished (failed runs are run again).
"""


def list_experiments() -> None:
    table = Table(title="Registered experiments")
    table.add_column("name")
    table.add_column("configs", justify="right")
    table.add_column("done", justify="right")
    for name, experiment in REGISTRY.items():
        done = existing_run_names(mlflow_experiment_names(experiment))
        n_done = sum(1 for cfg in experiment.configs if cfg["run_name"] in done)
        table.add_row(
            name, str(len(experiment.configs)), f"{n_done}/{len(experiment.configs)}"
        )
    console.print(table)


def list_configs(experiment_name: str) -> None:
    experiment = REGISTRY[experiment_name]
    done = existing_run_names(mlflow_experiment_names(experiment))
    table = Table(title=f"Configs for {experiment_name}")
    table.add_column("run_name")
    table.add_column("batch_size")
    table.add_column("model")
    table.add_column("optimizer")
    table.add_column("scheduler")
    table.add_column("split")
    table.add_column("obj_ids")
    table.add_column("status")
    for cfg in experiment.configs:
        train_conf = cfg["train"]
        scheduler_cls = train_conf["scheduler_cls"]
        table.add_row(
            cfg["run_name"],
            str(train_conf["batch_size"]),
            cfg["model"]["cls"].__name__,
            train_conf["optimizer_cls"].__name__,
            scheduler_cls.__name__ if scheduler_cls else "-",
            cfg["data"]["split"],
            str(cfg["data"]["obj_ids"]),
            "done" if cfg["run_name"] in done else "pending",
        )
    console.print(table)


def select_configs(
    configs: list[dict],
    done: set[str],
    interactive: bool,
    run_all: bool,
) -> list[dict]:
    if run_all:
        return configs

    if not interactive:
        pending = [cfg for cfg in configs if cfg["run_name"] not in done]
        console.print(
            f"Skipping {len(configs) - len(pending)} done, running {len(pending)} pending."
        )
        return pending

    choices = [
        questionary.Choice(
            title=f"{cfg['run_name']} ({'done' if cfg['run_name'] in done else 'pending'})",
            value=cfg["run_name"],
            checked=cfg["run_name"] not in done,
        )
        for cfg in configs
    ]
    selected_names = questionary.checkbox(
        "Select configs to run", choices=choices
    ).ask()
    if not selected_names:
        return []
    selected_names = set(selected_names)
    return [cfg for cfg in configs if cfg["run_name"] in selected_names]


def run_selected(configs: list[dict]) -> None:
    layout = DataLayout.from_env()
    for i, cfg in enumerate(configs, 1):
        console.print(f"[bold]Running[/bold] {cfg['run_name']} ({i}/{len(configs)})")
        run_experiment(cfg, layout)


def reset_runs(mlflow_names: set[str], run_names: set[str]) -> None:
    for exp_name in mlflow_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            continue
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
        for _, row in runs.iterrows():
            if row.get("tags.mlflow.runName") in run_names:
                mlflow.delete_run(str(row["run_id"]))
                console.print(f"Deleted run {row['tags.mlflow.runName']}")


# configs of every experiment are picked first, then everything runs unattended
def run_flow(experiment_names: list[str], interactive: bool, run_all: bool) -> None:
    configs = []
    for name in experiment_names:
        console.print(f"[bold cyan]Experiment: {name}[/bold cyan]")
        experiment = REGISTRY[name]
        done = existing_run_names(mlflow_experiment_names(experiment))
        configs += select_configs(
            experiment.configs, done, interactive=interactive, run_all=run_all
        )
    if not configs:
        console.print("Nothing to run.")
        return
    run_selected(configs)


def reset_flow(experiment_name: str) -> None:
    experiment = REGISTRY[experiment_name]
    mlflow_names = mlflow_experiment_names(experiment)
    done = existing_run_names(mlflow_names)
    if not done:
        console.print("No existing runs to reset.")
        return
    choices = [questionary.Choice(title=name, value=name) for name in sorted(done)]
    selected = questionary.checkbox("Select runs to delete", choices=choices).ask()
    if not selected:
        return
    reset_runs(mlflow_names, set(selected))


def cmd_list(args: argparse.Namespace) -> None:
    if args.experiment:
        list_configs(args.experiment)
    else:
        list_experiments()


def cmd_run(args: argparse.Namespace) -> None:
    run_flow(
        args.experiments or list(REGISTRY), interactive=args.select, run_all=args.all
    )


def cmd_reset(args: argparse.Namespace) -> None:
    reset_flow(args.experiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("experiment", nargs="?", choices=list(REGISTRY.keys()))
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run")
    # no experiments = all of them
    p_run.add_argument("experiments", nargs="*", choices=list(REGISTRY.keys()))
    p_run.add_argument("--select", action="store_true")
    p_run.add_argument("--all", action="store_true")
    p_run.set_defaults(func=cmd_run)

    p_reset = sub.add_parser("reset")
    p_reset.add_argument("experiment", choices=list(REGISTRY.keys()))
    p_reset.set_defaults(func=cmd_reset)

    return parser


def interactive_loop() -> None:
    while True:
        action = questionary.select(
            "Action", choices=["list", "run", "reset", "quit"]
        ).ask()
        if action is None or action == "quit":
            break

        if action == "list":
            experiment_name = questionary.select(
                "Experiment", choices=["(all)"] + list(REGISTRY.keys())
            ).ask()
            if experiment_name is None:
                continue
            if experiment_name == "(all)":
                list_experiments()
            else:
                list_configs(experiment_name)

        elif action == "run":
            experiment_names = questionary.checkbox(
                "Experiments to run", choices=list(REGISTRY.keys())
            ).ask()
            if not experiment_names:
                continue
            mode = questionary.select(
                "Mode",
                choices=["pending only", "select manually", "run all"],
            ).ask()
            if mode is None:
                continue
            run_flow(
                experiment_names,
                interactive=(mode == "select manually"),
                run_all=(mode == "run all"),
            )

        elif action == "reset":
            experiment_name = questionary.select(
                "Experiment", choices=list(REGISTRY.keys())
            ).ask()
            if experiment_name is None:
                continue
            reset_flow(experiment_name)


def mlflow_experiment_names(experiment: Experiment) -> set[str]:
    return {cfg["experiment_name"] for cfg in experiment.configs}


def existing_run_names(mlflow_names: set[str]) -> set[str]:
    names = set()
    for exp_name in mlflow_names:
        exp = mlflow.get_experiment_by_name(exp_name)
        if exp is None:
            continue
        runs = mlflow.search_runs(
            experiment_ids=[exp.experiment_id], filter_string="status = 'FINISHED'"
        )
        if runs.empty or "tags.mlflow.runName" not in runs.columns:
            continue
        names.update(runs["tags.mlflow.runName"].dropna())
    return names


def main() -> None:
    if len(sys.argv) == 1:
        interactive_loop()
        return
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
