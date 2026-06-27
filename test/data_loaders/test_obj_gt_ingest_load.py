# test/data_loaders/test_symmetry_ingestion.py
import os
import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

from data_loaders.ingest import ingest_symmetry_dataset
from data_loaders.dataload import load_batch_point_clouds_gt
from data_loaders.config import DataLoaderConfig


FIXTURES_DIR = Path(os.environ["FIXTURES_DIR"])

# TODO: deprectaed, ingestion logic will be rewrited
