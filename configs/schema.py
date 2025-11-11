from pydantic import BaseModel
from typing import Dict, List


class Config(BaseModel):
    project: str
    seed: int
    device: str
