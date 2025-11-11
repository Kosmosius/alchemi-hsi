import typer
import yaml
from pathlib import Path

from .utils.logging import get_logger
from .srf import SRFRegistry
from .training.seed import seed_everything
from .training.trainer import run_pretrain_mae, run_align, run_eval
from .data.validators import validate_dataset, validate_srf_dir

app = typer.Typer(add_completion=False)
_LOG = get_logger(__name__)


@app.command()
def validate_srf(root: str = "data/srf", sensor: str = "emit"):
    reg = SRFRegistry(root)
    srf = reg.get(sensor)
    ints = srf.row_integrals()
    _LOG.info("SRF integrals (first 5): %s", ints[:5])


@app.command()
def validate_data(config: str = "configs/data.yaml"):
    cfg = yaml.safe_load(Path(config).read_text())
    validate_dataset(cfg)
    validate_srf_dir(cfg.get("data", {}).get("srf_root", "data/srf"))


@app.command()
def pretrain_mae(config: str = "configs/train.mae.yaml"):
    seed_everything(42)
    run_pretrain_mae(config)


@app.command()
def align(config: str = "configs/train.align.yaml"):
    seed_everything(42)
    run_align(config)


@app.command()
def evaluate(config: str = "configs/eval.yaml"):
    run_eval(config)


if __name__ == "__main__":
    app()
