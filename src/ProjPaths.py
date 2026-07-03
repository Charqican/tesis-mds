from dataclasses import dataclass
from dotenv import load_dotenv
from pathlib import Path
import os

# simple path resolver
load_dotenv()

_DATA_ROOT = os.environ["DATA_ROOT"]
_DATA_GT_PLANE_SYMM_ORI = os.environ["DATA_GT_PLANE_SYMM_ORI"]
_DATA_GT_PLANE_SYMM = os.environ["DATA_GT_PLANE_SYMM"]
_DATA_PC_DATASET = os.environ["DATA_PC_DATASET"]
_DATA_FEATURES = os.environ["DATA_FEATURES"]
_DATA_INVARIANCE = os.environ["DATA_INVARIANCE"]


# Every file in these path expects a .npy format.
@dataclass
class ProjPath:
    root: str = _DATA_ROOT
    gt_plane_symm: str = _DATA_GT_PLANE_SYMM
    gt_plane_symm_ori: str = _DATA_GT_PLANE_SYMM_ORI
    pc_dataset: str = _DATA_PC_DATASET
    features: str = _DATA_FEATURES
    invariance: str = _DATA_INVARIANCE

    def get_path_feature(self, name: str | None = None) -> Path:
        if not name:
            return Path(self.features)

        new_feature_dir = Path(self.features) / name
        new_feature_dir.mkdir(parents=True, exist_ok=True)
        return new_feature_dir
