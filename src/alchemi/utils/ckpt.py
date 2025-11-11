import torch
from pathlib import Path
from typing import Dict, Any


def save_checkpoint(path: str | Path, state: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str | Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu")
