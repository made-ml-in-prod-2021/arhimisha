from dataclasses import dataclass
from typing import Dict, Any


@dataclass()
class ModelParams:
    model_name: str
    params: Dict[str, Any]
