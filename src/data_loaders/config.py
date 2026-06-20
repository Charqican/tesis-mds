from dataclasses import dataclass
from dotenv import load_dotenv
import os

# simple path resolver
load_dotenv()
_DEFAULT_RAW_DIR = os.environ["DATA_RAW"]
_DEFAULT_PROCESSED_DIR = os.environ["DATA_PROCESSED"]


@dataclass
class DataLoaderConfig:
    raw_dir: str = _DEFAULT_RAW_DIR
    processed_dir: str = _DEFAULT_PROCESSED_DIR
    mode: str = "separate"
    batch_size: int = 16
