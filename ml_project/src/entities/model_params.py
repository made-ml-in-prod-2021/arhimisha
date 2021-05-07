from dataclasses import dataclass
from typing import Dict, Any


@dataclass()
class ModelParams:
    name: str
    params: Dict[str, Any]
