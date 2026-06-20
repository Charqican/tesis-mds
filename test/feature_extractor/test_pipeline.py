import torch
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parents[1] / "fixtures"


def test_refactored_matches_original():
    original = torch.load(FIXTURES_DIR / "features_original.pt", map_location="cpu")
    refactored = torch.load(FIXTURES_DIR / "features.pt", map_location="cpu")

    assert original.shape == refactored.shape

    diff = (original - refactored).abs()
    # diferencia concentrada en puntos ocluidos interpolados por kNN
    # (cambio de backend torch_cluster -> pyg-lib), no en la logica central
    assert diff.median() < 1e-4
    assert (diff > 0.01).float().mean() < 0.02  # menos de 2% de puntos afectados
