from pose6d.loader import BOPLoader, LMOLoader, InstanceData


# Refactor: loader is necessary beause instance_uid resolves depenging of loader class.
def uids_by_visib_percentile(
    loader: BOPLoader,
    instances: list[tuple[int, int, InstanceData]],
    scene_id: int,
    percentiles: list[float],
    valid_uids: set[str] | None = None,
) -> dict[float, str]:
    valid = [
        (img_id, inst_idx, inst)
        for img_id, inst_idx, inst in instances
        if inst.visible_fract is not None
        and (
            valid_uids is None
            or loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
            in valid_uids
        )
    ]
    if not valid:
        return {}
    valid.sort(key=lambda x: x[2].visible_fract)

    n = len(valid)
    result = {}
    for p in percentiles:
        idx = round(p / 100 * (n - 1))
        idx = max(0, min(idx, n - 1))
        img_id, inst_idx, inst = valid[idx]
        result[p] = loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
    return result


def uids_by_visib_max(
    loader: BOPLoader,
    instances: list[tuple[int, int, InstanceData]],
    scene_id: int,
    max_visib: float,
    valid_uids: set[str] | None = None,
) -> list[str]:
    return [
        loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
        for img_id, inst_idx, inst in instances
        if inst.visible_fract is not None
        and inst.visible_fract <= max_visib
        and (
            valid_uids is None
            or loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
            in valid_uids
        )
    ]


def uids_by_visib_min(
    loader: BOPLoader,
    instances: list[tuple[int, int, InstanceData]],
    scene_id: int,
    min_visib: float,
    valid_uids: set[str] | None = None,
) -> list[str]:
    return [
        loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
        for img_id, inst_idx, inst in instances
        if inst.visible_fract is not None
        and inst.visible_fract >= min_visib
        and (
            valid_uids is None
            or loader.instance_uid(scene_id, img_id, inst.obj_id, inst_idx)
            in valid_uids
        )
    ]
