import os

import pytest
from typing import List

from src.entities import FeatureParams, SplittingParams, ModelParams


@pytest.fixture()
def dataset_path():
    curdir = os.path.dirname(__file__)
    return os.path.join(curdir, "test_data_sample.csv")


@pytest.fixture()
def target_feature() -> List[str]:
    return ["target"]


@pytest.fixture()
def categorical_features() -> List[str]:
    return [
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture
def numerical_features() -> List[str]:
    return [
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ]


@pytest.fixture
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        target_feature: List[str],
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        target_feature=target_feature,
    )
    return params


@pytest.fixture()
def splitting_params() -> SplittingParams:
    return SplittingParams(
        val_size=0.1,
        test_size=0.1,
        random_state=42
    )


@pytest.fixture()
def model_params() -> ModelParams:
    return ModelParams(
        model_name="LinearSVC",
        params={
            "penalty": "l2",
            "loss": "squared_hinge",
            "dual": False,
            "tol": 0.0001,
            "C": 0.5,
            "multi_class": "ovr",
            "fit_intercept": True,
        }
    )
