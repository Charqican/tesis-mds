from dataclasses import dataclass
from dotenv import load_dotenv
import os

# simple path resolver


@dataclass
class DataLoaderConfig:
    mode: str = "separate"
    batch_size: int = 16
