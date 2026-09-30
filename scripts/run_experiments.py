import argparse
import sys
from dotenv import load_dotenv
import questionary
from rich.console import Console
from rich.table import Table

from experiments.registry import REGISTRY, Experiment
from experiments.mlflow_wrapper import run_experiment

load_dotenv()  # this reads the .env mlflow variable before importing mlflow.
import mlflow


console = Console()
"""
The following script is a Rich cli program with the only purpose of simplify the 
mlflow artifacts manangment and experiments execution. 
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
    table.add_column("optimizer")
    table.add_column("scheduler")
    table.add_column("obj_ids")
    table.add_column("status")
    for cfg in experiment.configs:
        setup_conf = cfg["setup_conf"]
        train_conf = cfg["train_conf"]
        status = "done" if cfg["run_name"] in done else "pending"
        scheduler_cls = train_conf.get("scheduler_cls")
        table.add_row(
            cfg["run_name"],
            str(setup_conf.get("batch_size", "")),
            train_conf["optimizer_cls"].__name__,
            scheduler_cls.__name__ if scheduler_cls else "-",
            str(sorted(setup_conf.get("sel_obj_ids", []))),
            status,
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
            f"Skipping {len(configs) - len(pending)} already done, running {len(pending)} pending."
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


def run_selected(experiment: Experiment, configs: list[dict]) -> None:
    for cfg in configs:
        console.print(f"[bold]Running[/bold] {cfg['run_name']}")
        run_experiment(cfg, experiment.setup_func, experiment.train_eng)


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


def run_flow(experiment_name: str, interactive: bool, run_all: bool) -> None:
    experiment = REGISTRY[experiment_name]
    done = existing_run_names(mlflow_experiment_names(experiment))
    configs = select_configs(
        experiment.configs, done, interactive=interactive, run_all=run_all
    )
    if not configs:
        console.print("Nothing to run.")
        return
    run_selected(experiment, configs)


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
    run_flow(args.experiment, interactive=args.select, run_all=args.all)


def cmd_reset(args: argparse.Namespace) -> None:
    reset_flow(args.experiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="experiments-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("experiment", nargs="?", choices=list(REGISTRY.keys()))
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run")
    p_run.add_argument("experiment", choices=list(REGISTRY.keys()))
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
            "Accion", choices=["list", "run", "reset", "salir"]
        ).ask()
        if action is None or action == "salir":
            break

        if action == "list":
            experiment_name = questionary.select(
                "Experimento", choices=["(todos)"] + list(REGISTRY.keys())
            ).ask()
            if experiment_name is None:
                continue
            if experiment_name == "(todos)":
                list_experiments()
            else:
                list_configs(experiment_name)

        elif action == "run":
            experiment_name = questionary.select(
                "Experimento", choices=list(REGISTRY.keys())
            ).ask()
            if experiment_name is None:
                continue
            mode = questionary.select(
                "Modo",
                choices=["solo pendientes", "seleccionar manualmente", "correr todo"],
            ).ask()
            if mode is None:
                continue
            run_flow(
                experiment_name,
                interactive=(mode == "seleccionar manualmente"),
                run_all=(mode == "correr todo"),
            )

        elif action == "reset":
            experiment_name = questionary.select(
                "Experimento", choices=list(REGISTRY.keys())
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
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
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
