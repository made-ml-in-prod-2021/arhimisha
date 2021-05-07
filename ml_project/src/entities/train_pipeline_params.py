import yaml
from dataclasses import dataclass
from typing import List
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplittingParams
from .model_params import ModelParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    metric_path: str
    feature_params: FeatureParams
    splitting_params: SplittingParams
    model_params: List[ModelParams]


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    schema = class_schema(TrainingPipelineParams)
    with open(path, "r") as input_stream:
        return schema.load(yaml.safe_load(input_stream))
