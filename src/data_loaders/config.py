from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    raw_dir: str = "../data/raw"
    processed_dir: str = "../data/processed"
    mode: str = "separate"
    batch_size: int = 16
