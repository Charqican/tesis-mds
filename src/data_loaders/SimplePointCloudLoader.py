from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, TypeVar, Generic, cast

import numpy as np
import torch
from pytorch3d.io import IO
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch

from data_loaders.config import DataLoaderConfig
from data_loaders.utils import find_files
from logger import dataload_logger

T = TypeVar("T")

SymmetryPlanes = list[tuple[torch.Tensor, torch.Tensor]]


class BaseLoader(ABC, Generic[T]):
    extension: str
    GT_EXTENSION = "txt"

    def __init__(
        self,
        config: DataLoaderConfig,
        input_path: str | Path,
        n: int | None = None,
        sort: bool = False,
        gt_path: str | Path | None = None,
    ):
        """
        param gt_path: carpeta donde buscar los .txt de ground truth
            (planos de simetria). Valor None asume mismo archivo de objetos
        """
        self.input_path = Path(input_path)
        self.gt_path = Path(gt_path) if gt_path is not None else self.input_path
        self.config = config
        self.files = find_files(self.input_path, self.extension, n=n, sort=sort)
        dataload_logger.info(
            f"{type(self).__name__}: found {len(self.files)} objects in {self.input_path} \n"
        )
        self._gt_files_by_stem: dict[str, Path] | None = None

    def __len__(self) -> int:
        return len(self.files)

    @abstractmethod
    def _load(self, file: Path) -> T: ...

    def _index_gt_files(self) -> dict[str, Path]:
        """
        Indexa los archivos GT disponibles en `gt_path` por stem (nombre sin extension)
        """
        if self._gt_files_by_stem is None:
            gt_files = find_files(self.gt_path, self.GT_EXTENSION)
            self._gt_files_by_stem = {f.stem: f for f in gt_files}
        return self._gt_files_by_stem

    @staticmethod
    def _parse_symmetry_planes(file: Path) -> SymmetryPlanes:
        """
        Parsea un archivo .txt de planos de simetria ground truth.

        Formato esperado:
            <numero de planos>
            plane a, b, c, d, e, f
            ... (una linea por plano, hasta 3 lineas)

        (a, b, c): vector normal del plano. (d, e, f): un punto contenido
        en el plano.

        return: lista de (normal, point), cada uno tensor (3,)
        """
        lines = [line for line in file.read_text().strip().splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"{file}: GT empty")

        n_planes = int(lines[0].strip())
        plane_lines = lines[1 : 1 + n_planes]
        if len(plane_lines) != n_planes:
            raise ValueError(
                f"{file}: expected {n_planes} but only {len(plane_lines)} lines were given"
            )

        planes: SymmetryPlanes = []
        for line in plane_lines:
            _, values_str = line.strip().split(maxsplit=1)
            values = [float(v.strip()) for v in values_str.split(" ")]
            if len(values) != 6:
                raise ValueError(
                    f"{file}: expected 6 values (normal + point), but only {len(values)} were found"
                )
            normal = torch.tensor(values[:3], dtype=torch.float32)
            point = torch.tensor(values[3:], dtype=torch.float32)
            planes.append((normal, point))

        return planes

    def _load_gt_for_chunk(self, chunk: list[Path]) -> list[SymmetryPlanes]:
        """
        Carga los planos GT correspondientes a cada archivo de `chunk`,
        emparejando por nombre (stem).
        """
        gt_by_stem = self._index_gt_files()
        result = []
        for f in chunk:
            gt_file = gt_by_stem.get(f.stem)
            if gt_file is None:
                raise FileNotFoundError(
                    f"GT not found (.{self.GT_EXTENSION}) for '{f.stem}' in {self.gt_path}"
                )
            result.append(self._parse_symmetry_planes(gt_file))
        return result

    def __iter__(self) -> Iterator[tuple[str, T]]:
        for f in self.files:
            yield f.stem, self._load(f)

    @abstractmethod
    def batches(
        self, gt: bool = False
    ) -> (
        Iterator[tuple[list[str], T]]
        | Iterator[tuple[list[str], list[SymmetryPlanes], T]]
    ): ...


class PointCloudLoader(BaseLoader[torch.Tensor]):
    """
    Carga lazy de point clouds desde una carpeta con un .npy por objeto.
    No carga nada en memoria hasta que se itera.

    Asume que todos los archivos tienen el mismo numero de puntos P, lo
    que permite apilarlos directamente con torch.stack en `batches()`.
    n: numero de archivos que buscar en carpeta. Si es None busca todos.
    """

    extension = "npy"

    def _load(self, file: Path) -> torch.Tensor:
        return torch.from_numpy(np.load(file))

    def batches(
        self, gt: bool = False
    ) -> (
        Iterator[tuple[list[str], torch.Tensor]]
        | Iterator[tuple[list[str], list[SymmetryPlanes], torch.Tensor]]
    ):
        """
        Yields (names, point_clouds) o, si gt=True, (names, symmetry_planes, point_clouds).
        names: list[str]
        point_clouds: Tensor (b, P, 3) donde b <= batch_size
            (el ultimo batch puede ser mas chico si len(self) no es multiplo)
        symmetry_planes: list de largo b, cada elemento es la lista de
            (normal, point) del objeto correspondiente
        """
        batch_size = self.config.batch_size
        for i in range(0, len(self.files), batch_size):
            chunk = self.files[i : i + batch_size]
            names = [f.stem for f in chunk]
            point_clouds = torch.stack([self._load(f) for f in chunk], dim=0)

            if gt:
                symmetry_planes = self._load_gt_for_chunk(chunk)
                yield names, symmetry_planes, point_clouds
            else:
                yield names, point_clouds


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
        gt_path: ver BaseLoader.__init__. Por default, los .txt GT se
            buscan en `input_path` (donde tambien estan los .obj), que
            coincide con lo descrito: "los gt tienen el mismo nombre que
            los .obj y estan en la misma carpeta".
    """

    def __init__(
        self,
        config: DataLoaderConfig,
        input_path: str | Path,
        n: int | None = None,
        sort: bool = False,
        extension: str = "obj",
        device: str = "cpu",
        gt_path: str | Path | None = None,
    ):
        self.extension = extension
        self.device = device
        self._io = IO()
        super().__init__(config, input_path, n=n, sort=sort, gt_path=gt_path)

    def _load(self, file: Path) -> Meshes:
        return self._io.load_mesh(file, device=self.device)

    def batches(
        self, gt: bool = False
    ) -> (
        Iterator[tuple[list[str], Meshes]]
        | Iterator[tuple[list[str], list[SymmetryPlanes], Meshes]]
    ):
        """
        Yields (names, mesh_batch) o, si gt=True, (names, symmetry_planes, mesh_batch).
        names: list[str], mesh_batch: Meshes con b <= batch_size.
        """
        batch_size = self.config.batch_size
        for i in range(0, len(self.files), batch_size):
            chunk = self.files[i : i + batch_size]
            names = [f.stem for f in chunk]
            meshes = join_meshes_as_batch([self._load(f) for f in chunk])

            if gt:
                symmetry_planes = self._load_gt_for_chunk(chunk)
                yield names, symmetry_planes, meshes
            else:
                yield names, meshes

    @staticmethod
    def to_point_cloud(mesh: Meshes, num_points: int) -> torch.Tensor:
        """
        Sampleo uniforme de puntos sobre la superficie del mesh.
        """
        samples = sample_points_from_meshes(mesh, num_points)
        return cast(torch.Tensor, samples)
