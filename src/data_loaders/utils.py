from pathlib import Path


def find_files(
    file_dir: Path, file_extension: str, n: int | None = None, sort: bool = False
) -> list[Path]:
    if sort:
        files = sorted(file_dir.glob(f"*.{file_extension}"))  # buscar arhivos .parquet
    else:
        files = list(file_dir.glob(f"*.{file_extension}"))
    return files[:n]
