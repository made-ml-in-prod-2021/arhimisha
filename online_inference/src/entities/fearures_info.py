import yaml
from dataclasses import dataclass
from typing import List
from marshmallow_dataclass import class_schema


@dataclass()
class FeaturesInfo:
    features_number: int
    feature_names: List[str]


def read_features_info(path: str) -> FeaturesInfo:
    FeaturesInfoSchema = class_schema(FeaturesInfo)
    schema = FeaturesInfoSchema()
    with open(path, "r") as input_stream:
        return schema.load(yaml.safe_load(input_stream))
