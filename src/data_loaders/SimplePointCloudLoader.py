from pathlib import Path

import numpy as np
import torch

from data_loaders.config import DataLoaderConfig
from data_loaders.utils import find_files
from logger import dataload_logger


class PointCloudLoader:
    """
    Carga lazy de point clouds desde una carpeta con un .npy por objeto.
    No carga nada en memoria hasta que se itera.

    Modos de uso:
        for name, pc in loader:                  # un objeto a la vez, (P, 3)
        for names, batch in loader.batches(16):   # de a 16, (16, P, 3)
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        n: int | None = None,
        sort: bool = True,
    ):
        self.config = config
        processed_dir = Path(config.processed_dir)
        self.files = find_files(processed_dir, "npy", n=n, sort=sort)
        dataload_logger.info(
            f"PointCloudLoader: found {len(self.files)} objects in {processed_dir}"
        )

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self):
        for f in self.files:
            point_cloud = torch.from_numpy(np.load(f))  # (P, 3)
            yield f.stem, point_cloud

    def batches(self):
        """
        Yields (names, point_clouds) en grupos de batch_size.
        names: list[str], point_clouds: Tensor (b, P, 3) donde b <= batch_size
        (el ultimo batch puede ser mas chico si len(self) no es multiplo).
        """
        batch_size = self.config.batch_size
        for i in range(0, len(self.files), batch_size):
            chunk = self.files[i : i + batch_size]

            names = []
            point_clouds = []
            for f in chunk:
                names.append(f.stem)
                point_clouds.append(torch.from_numpy(np.load(f)))

            yield names, torch.stack(point_clouds, dim=0)  # (b, P, 3)
