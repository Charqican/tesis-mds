import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

"""

This file exist because there were a lot of paths to manage between server - local. 


This is the Layout expected and used by every other object:
    {root}/{dataset}/cache/{pT}/points_pT/{uid}.npz
    {root}/{dataset}/cache/{pT}/manifest.json
    {root}/experiments/{experiment}/input/{uid}.npz      (dgedi features)
    {root}/experiments/{experiment}/target/{uid}.npz     (symmetry field)
    {root}/experiments/{experiment}/manifest.json
    {root}/experiments/{experiment}/dgedi_inputs.txt     (pending dgedi work list)

- DataLayout is very similar to loader.py, as it resolves paths and abstract the logic.
- PTCofing and SymmConfig parses tomls configurations. Their use is restricted to dataset creation and verisioning
"""

DATASETS = ("lmo", "pbr")
DGEDI_DIMS = (32, 64)

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PT_TOML = CONFIGS_DIR / "pT_extract.toml"
SYMM_TOML = CONFIGS_DIR / "symm_features.toml"


def uid_dataset(uid: str) -> str:
    return uid.split("_", 1)[0]


@dataclass(frozen=True)
class DataLayout:
    root: Path

    @classmethod
    def from_env(cls, env_var: str = "POSE6D_ROOT") -> "DataLayout":
        root = os.environ.get(env_var)
        if not root:
            raise ValueError(f"Environment variable {env_var} not set")
        return cls(Path(root))

    # --- pT cache (per dataset) ---
    def pT_dir(self, dataset: str, pT: str) -> Path:
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}, expected one of {DATASETS}")
        return self.root / dataset / "cache" / pT

    def points_dir(self, dataset: str, pT: str) -> Path:
        return self.pT_dir(dataset, pT) / "points_pT"

    def pT_manifest(self, dataset: str, pT: str) -> Path:
        return self.pT_dir(dataset, pT) / "manifest.json"

    def points_path(self, pT: str, uid: str) -> Path:
        return self.points_dir(uid_dataset(uid), pT) / f"{uid}.npz"

    # --- experiments (datasets mixed, uids are prefixed by dataset) ---
    def experiment_dir(self, experiment: str) -> Path:
        return self.root / "experiments" / experiment

    def input_dir(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "input"

    def target_dir(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "target"

    def experiment_manifest(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "manifest.json"

    def dgedi_inputs(self, experiment: str) -> Path:
        return self.experiment_dir(experiment) / "dgedi_inputs.txt"


# accepts formats like: 3, [2, 3], "0-49" or "0-9,12"
def parse_scenes(value: int | str | list[int]) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value,)
    if isinstance(value, list):
        return tuple(int(v) for v in value)
    scenes = []
    for part in value.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            scenes.extend(range(int(start), int(end) + 1))
        else:
            scenes.append(int(part))
    return tuple(scenes)


@dataclass(frozen=True)
class PTConfig:
    name: str
    min_visib: float
    n_points: int
    outliers: dict | None
    scenes: dict[str, tuple[int, ...]]  # dataset -> scene ids

    # what must match between the toml and an existing version on disk
    def params(self) -> dict:
        return {
            "min_visib": self.min_visib,
            "n_points": self.n_points,
            "outliers": dict(self.outliers) if self.outliers else None,
        }


@dataclass(frozen=True)
class SymmConfig:
    name: str
    pT: str
    dgedi_dim: int
    canonical_samples: int = 20000
    seed: int = 0

    def target_params(self) -> dict:
        return {"canonical_samples": self.canonical_samples, "seed": self.seed}


def load_pT_configs(path: Path = PT_TOML) -> dict[str, PTConfig]:
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)

    configs = {}
    for name, section in raw.items():
        scenes = {}
        for source in section.get("sources", []):
            dataset = source["dataset"]
            if dataset not in DATASETS:
                raise ValueError(f"[{name}] unknown dataset {dataset!r}")
            if dataset in scenes:
                raise ValueError(f"[{name}] dataset {dataset!r} declared twice")
            scenes[dataset] = parse_scenes(source["scenes"])
        if not scenes:
            raise ValueError(f"[{name}] needs at least one [[{name}.sources]] entry")

        outliers = section.get("outliers")
        if outliers is not None:
            outliers = {
                "nb_neighbors": int(outliers["nb_neighbors"]),
                "std": float(outliers["std"]),
            }
        configs[name] = PTConfig(
            name=name,
            min_visib=float(section["min_visib"]),
            n_points=int(section["n_points"]),
            outliers=outliers,
            scenes=scenes,
        )
    return configs


def load_symm_configs(
    path: Path = SYMM_TOML, pT_configs: dict[str, PTConfig] | None = None
) -> dict[str, SymmConfig]:
    with Path(path).open("rb") as f:
        raw = tomllib.load(f)

    configs = {}
    for name, section in raw.items():
        cfg = SymmConfig(name=name, **section)
        if cfg.dgedi_dim not in DGEDI_DIMS:
            raise ValueError(f"[{name}] dgedi_dim must be one of {DGEDI_DIMS}")
        if pT_configs is not None and cfg.pT not in pT_configs:
            raise ValueError(
                f"[{name}] pT {cfg.pT!r} is not defined in pT_extract.toml"
            )
        configs[name] = cfg
    return configs
