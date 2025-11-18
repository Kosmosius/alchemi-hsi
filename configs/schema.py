from pydantic import BaseModel


class Config(BaseModel):  # type: ignore[misc]
    project: str
    seed: int
    device: str
