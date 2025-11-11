from pathlib import Path


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, data: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(data, encoding="utf-8")
