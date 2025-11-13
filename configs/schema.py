from pydantic import BaseModel


class Config(BaseModel):
    project: str
    seed: int
    device: str
