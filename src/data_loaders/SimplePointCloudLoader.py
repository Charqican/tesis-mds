from pathlib import Path

import numpy as np
import torch

from ProjPaths import ProjPath
from data_loaders.config import DataLoaderConfig
from data_loaders.utils import find_files
from logger import dataload_logger


class PointCloudLoader:
    """
    Carga lazy de point clouds desde una carpeta con un .npy por objeto.
    No carga nada en memoria hasta que se itera.
    implementa utilid para cargar features

    Args:
        config: A DataLoaderConfig object
        n : the total number of files to load. if nothing is passed load every file in config path
        sort: sort files before serving.
        input_path: Donde se espera encontrar los archivos en formato .npy. Se asume que cada .npy es una nube de puntos
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        input_path: str | Path,
        n: int | None = None,
        sort: bool = False,
    ):
        self.input_path = Path(input_path)
        self.config = config
        self.files = find_files(self.input_path, "npy", n=n, sort=sort)
        # remove faulty file
        # self.files.remove(self.input_path / "895563d304772f50ad5067eac75a07f7.npy")
        dataload_logger.info(
            f"PointCloudLoader: found {len(self.files)} objects in {self.input_path} \n"
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
