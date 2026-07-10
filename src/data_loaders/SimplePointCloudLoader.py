from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TypeVar, Generic, cast

import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch

from ProjPaths import ProjPath
from data_loaders.config import DataLoaderConfig
from data_loaders.utils import find_files
from logger import dataload_logger

T = TypeVar("T")


class BaseLoader(ABC, Generic[T]):
    extension: str

    def __init__(
        self,
        config: DataLoaderConfig,
        input_path: str | Path,
        n: int | None = None,
        sort: bool = False,
    ):
        self.input_path = Path(input_path)
        self.config = config
        self.files = find_files(self.input_path, self.extension, n=n, sort=sort)
        dataload_logger.info(
            f"{type(self).__name__}: found {len(self.files)} objects in {self.input_path} \n"
        )

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def _load(self, file: Path) -> T: ...

    def __iter__(self):
        for f in self.files:
            yield f.stem, self._load(f)

    @abstractmethod
    def batches(self) -> Iterator[tuple[list[str], T]]: ...


class PointCloudLoader(BaseLoader[torch.Tensor]):
    """
    Carga lazy de point clouds desde una carpeta con un .npy por objeto.
    No carga nada en memoria hasta que se itera.

    Asume que todos los archivos tienen el mismo numero de puntos P, lo
    que permite apilarlos directamente con torch.stack en `batches()`.
    """

    extension = "npy"

    def _load(self, file: Path) -> torch.Tensor:  # (P, 3)
        return torch.from_numpy(np.load(file))

    def batches(self) -> Iterator[tuple[list[str], torch.Tensor]]:
        """
        Yields (names, point_clouds) en grupos de batch_size.
        names: list[str], point_clouds: Tensor (b, P, 3) donde b <= batch_size
        (el ultimo batch puede ser mas chico si len(self) no es multiplo).
        """
        batch_size = self.config.batch_size
        for i in range(0, len(self.files), batch_size):
            chunk = self.files[i : i + batch_size]

            names = [f.stem for f in chunk]
            point_clouds = [self._load(f) for f in chunk]

            yield names, torch.stack(point_clouds, dim=0)  # (b, P, 3)


class MeshLoader(BaseLoader[Meshes]):
    """
    Carga lazy de meshes (vertices + caras) desde una carpeta con un
    archivo por objeto. No carga nada en memoria hasta que se itera.

    A diferencia de PointCloudLoader, los meshes tienen topologia
    heterogenea (distinto numero de vertices/caras por objeto), por lo
    que el batching se delega a `Meshes` de pytorch3d en vez de
    torch.stack.

    Args:
        config: A DataLoaderConfig object
        input_path: carpeta donde se espera encontrar los archivos
        n: numero total de archivos a cargar. Si no se pasa, carga todos
            los archivos encontrados en input_path
        sort: ordenar archivos antes de servirlos
        extension: formato de archivo de malla ("obj", "ply", etc).
            Default "obj" (formato tipico de ShapeNet).
        device: device donde se cargan los tensores de la malla.
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        input_path: str | Path,
        n: int | None = None,
        sort: bool = False,
        extension: str = "obj",
        device: str = "cpu",
    ):
        self.extension = extension
        self.device = device
        self._io = IO()
        super().__init__(config, input_path, n=n, sort=sort)

    def _load(self, file: Path) -> Meshes:
        return self._io.load_mesh(file, device=self.device)

    def batches(self):
        """
        Yields (names, mesh_batch) en grupos de batch_size.
        names: list[str], mesh_batch: Meshes con b <= batch_size mallas
        (pytorch3d soporta nativamente topologia heterogenea al batchear,
        no requiere que todas las mallas tengan el mismo #vertices/#caras).
        """
        batch_size = self.config.batch_size
        for i in range(0, len(self.files), batch_size):
            chunk = self.files[i : i + batch_size]

            names = [f.stem for f in chunk]
            meshes = [self._load(f) for f in chunk]

            yield names, join_meshes_as_batch(meshes)

    @staticmethod
    def to_point_cloud(mesh: Meshes, num_points: int) -> torch.Tensor:
        """
        Sampleo uniforme de puntos sobre la superficie del mesh.

        """
        samples = sample_points_from_meshes(mesh, num_points)
        # sample_points_from_meshes tiene firma Union segun return_normals/
        # return_textures; con ambos en False (default) siempre es un Tensor.
        return cast(torch.Tensor, samples)  # (b, num_points, 3)
