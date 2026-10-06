import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from pose6d.layout import (
    PT_TOML,
    SYMM_TOML,
    DataLayout,
    PTConfig,
    SymmConfig,
    load_pT_configs,
    load_symm_configs,
    parse_scenes,
    uid_dataset,
)
from pose6d.loader import BOPLoader, loader_from_env
from pose6d.preprocessing import (
    ExtractParams,
    compute_targets,
    extract_scene,
    extract_uids,
)
from logger import scripts_extraction_logger as log

"""
Generates the processed data described in configs/pT_extract.toml (pT)
and configs/symm_features.toml (symmetry field). dgedi runs in its own environment,
so this script writes the work list and prints the exact command to run.

# Use:

    # run every configuration in both toml 
    python scripts/pose6d_prepare_data.py

    # one pT version / one preprocessing version 
    python scripts/pose6d_prepare_data.py --config-name rm_outliers_20_2_visib_10
    python scripts/pose6d_prepare_data.py --version-name scalarfield_exp3

    # specific cases with the parameters of a version. This is used mainfly for analysis
    python scripts/pose6d_prepare_data.py --version-name scalarfield_exp3 --scenes pbr:3,5-7 lmo:2
    python scripts/pose6d_prepare_data.py --version-name scalarfield_exp3 --uids pbr_scene000003_img000012_obj000010_inst01
"""

EXTERNAL_DIR = Path(__file__).resolve().parents[1] / "external"


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def write_manifest(path: Path, manifest: dict) -> None:
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


# A scene listed in the manifest file is skipped
def run_pT(
    layout: DataLayout,
    cfg: PTConfig,
    scenes: dict[str, tuple[int, ...]] | None,
    uids: list[str] | None,
) -> None:
    params = ExtractParams(cfg.n_points, cfg.min_visib, cfg.outliers)
    datasets = {uid_dataset(u) for u in uids} if uids else (scenes or cfg.scenes)

    for dataset in datasets:
        loader = loader_from_env(dataset)
        obj_ids = loader.symmetric_obj_ids()
        out_dir = layout.points_dir(dataset, cfg.name)
        manifest_path = layout.pT_manifest(dataset, cfg.name)
        manifest = read_manifest(manifest_path)
        manifest.update({"pT": cfg.name, "params": cfg.params()})
        manifest.setdefault("scenes", {})
        manifest.setdefault("uids", [])

        if uids:
            pending = [
                u
                for u in uids
                if uid_dataset(u) == dataset and not (out_dir / f"{u}.npz").exists()
            ]
            saved = extract_uids(loader, pending, obj_ids, params, out_dir)
            log.info(f"pT '{cfg.name}' ({dataset}): {len(saved)} uids extracted")
            manifest["uids"] = sorted(set(manifest["uids"]) | set(saved))
            write_manifest(manifest_path, manifest)
            continue

        for scene_id in (scenes or cfg.scenes)[dataset]:
            if str(scene_id) in manifest["scenes"]:
                log.info(f"{dataset} scene {scene_id}: already extracted, skipping")
                continue
            if not loader.paths.scene_dir(scene_id).exists():
                log.warning(f"{dataset} scene {scene_id}: not found on disk, skipping")
                continue
            stats = extract_scene(loader, scene_id, obj_ids, params, out_dir)
            log.info(f"{dataset} scene {scene_id}: {stats}")
            manifest["scenes"][str(scene_id)] = stats
            write_manifest(manifest_path, manifest)


def run_for_version(
    layout: DataLayout,
    exp: SymmConfig,
    pT_cfg: PTConfig,
    scenes: dict[str, tuple[int, ...]] | None,
    uids: list[str] | None,
) -> str | None:
    """Computes missing targets and returns the dgedi command if inputs are pending."""
    if uids:
        points = {u: layout.points_path(exp.pT, u) for u in uids}
        points = {u: p for u, p in points.items() if p.exists()}
    else:
        points = {}
        for dataset, wanted in (scenes or pT_cfg.scenes).items():
            for path in sorted(layout.points_dir(dataset, exp.pT).glob("*.npz")):
                if not scenes or BOPLoader.parse_instance_uid(path.stem)[1] in wanted:
                    points[path.stem] = path

    target_dir = layout.target_dir(exp.name)
    input_dir = layout.input_dir(exp.name)
    pending_targets = {
        u: p for u, p in points.items() if not (target_dir / f"{u}.npz").exists()
    }
    log.info(
        f"'{exp.name}': {len(points)} uids, {len(pending_targets)} targets pending"
    )
    compute_targets(pending_targets, target_dir, exp.canonical_samples, exp.seed)

    write_manifest(
        layout.version_manifest(exp.name),
        {
            "version": exp.name,
            "pT": exp.pT,
            "pT_params": pT_cfg.params(),
            "dgedi_dim": exp.dgedi_dim,
            "target_params": exp.target_params(),
        },
    )

    pending_inputs = [
        p for u, p in points.items() if not (input_dir / f"{u}.npz").exists()
    ]
    if not pending_inputs:
        return None
    work_list = layout.dgedi_inputs(exp.name)
    work_list.write_text("\n".join(str(p) for p in pending_inputs) + "\n")
    return (
        f"# {exp.name}: {len(pending_inputs)} instances\n"
        f"{EXTERNAL_DIR / 'dgedi_env/bin/python'} {EXTERNAL_DIR / 'extract_dgedi_features.py'} "
        f"--inputs-list {work_list} --output-dir {input_dir} --dim {exp.dgedi_dim}"
    )


# "pbr:3,5-7" -> {"pbr": (3, 5, 6, 7)}
def parse_scene_overrides(tokens: list[str]) -> dict[str, tuple[int, ...]]:
    scenes = {}
    for token in tokens:
        dataset, _, spec = token.partition(":")
        scenes[dataset] = parse_scenes(spec)
    return scenes


def main() -> None:
    load_dotenv()
    args = parse_args()

    layout = DataLayout.from_env()
    pT_configs = load_pT_configs(args.pT_toml)
    symm_configs = load_symm_configs(args.symm_toml, pT_configs)
    scenes = parse_scene_overrides(args.scenes) if args.scenes else None

    if args.config_name:
        run_pT(layout, pT_configs[args.config_name], scenes, args.uids)
        return

    if args.version_name:
        versions = [symm_configs[args.version_name]]
        pT_names = [versions[0].pT]
    else:
        versions = list(symm_configs.values())
        pT_names = list(pT_configs)

    for name in pT_names:
        run_pT(layout, pT_configs[name], scenes, args.uids)

    commands = [
        run_for_version(layout, exp, pT_configs[exp.pT], scenes, args.uids)
        for exp in versions
    ]
    commands = [c for c in commands if c]
    if commands:
        print("\nPending dgedi features, run:\n\n" + "\n\n".join(commands))
    else:
        print("\nNo dgedi features pending.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate pT, symmetry targets and the dgedi work list from the data tomls."
    )
    names = p.add_mutually_exclusive_group()
    names.add_argument(
        "--config-name", help="Only extract this pT version (pT_extract.toml section)."
    )
    names.add_argument(
        "--version-name",
        help="Only process this version (symm_features.toml section) and its pT version.",
    )

    subset = p.add_mutually_exclusive_group()
    subset.add_argument(
        "--scenes",
        nargs="+",
        metavar="DATASET:SCENES",
        help="Only these scenes, e.g. pbr:3,5-7 lmo:2. Needs --config-name or --version-name.",
    )
    subset.add_argument(
        "--uids",
        nargs="+",
        help="Only these instance uids. Needs --config-name or --version-name.",
    )

    p.add_argument("--pT-toml", type=Path, default=PT_TOML)
    p.add_argument("--symm-toml", type=Path, default=SYMM_TOML)

    args = p.parse_args()
    if (args.scenes or args.uids) and not (args.config_name or args.version_name):
        p.error("--scenes/--uids need --config-name or --version-name")
    return args


if __name__ == "__main__":
    main()
