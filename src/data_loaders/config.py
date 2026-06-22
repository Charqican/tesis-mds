from dataclasses import dataclass
from dotenv import load_dotenv
import os

# simple path resolver
load_dotenv()
_DEFAULT_RAW_DIR = os.environ["DATA_RAW"]
_DEFAULT_PROCESSED_DIR = os.environ["DATA_PROCESSED"]
_DEFAULT_SYMMETRY_DIR = os.environ["DATA_SYM_PLANE"]
_DEFAULT_SYMMETRY_PROCESSED_DIR = os.environ["DATA_SYM_PLANE_PROCESSED"]


@dataclass
class DataLoaderConfig:
    raw_dir: str = _DEFAULT_RAW_DIR
    processed_dir: str = _DEFAULT_PROCESSED_DIR
    symmetry_obj_dir: str = _DEFAULT_SYMMETRY_DIR
    symmetry_processed_dir: str = _DEFAULT_SYMMETRY_PROCESSED_DIR
    mode: str = "separate"
    batch_size: int = 16
