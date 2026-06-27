import numpy as np
import torch
import pytest
from pathlib import Path
from data_loaders.dataload import load_batch_point_clouds, load_labels
from data_loaders.config import DataLoaderConfig

# Deprecated, will be rewrited in the future
