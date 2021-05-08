from typing import List

import numpy as np
import pandas as pd
import pytest

from src.features_transformer.build_features_transformer import build_transformer
from src.entities.feature_params import FeatureParams


@pytest.fixture()
def categorical_feature() -> str:
    return "categorical_feature"


@pytest.fixture()
def categorical_values() -> List[str]:
    return ["cat", "dog", "cow", np.nan]


@pytest.fixture()
def numerical_feature() -> str:
    return "numerical_feature"


@pytest.fixture()
def numerical_values() -> np.ndarray:
    return np.random.random(4) * 10 + 100


# @pytest.fixture()
# def categorical_values_with_nan(categorical_values: List[str]) -> List[str]:
#     return categorical_values + [np.nan]


@pytest.fixture
def fake_data(
        categorical_feature: str, categorical_values: List[str],
        numerical_feature: str, numerical_values: np.ndarray
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            categorical_feature: categorical_values,
            numerical_feature: numerical_values
        }
    )


@pytest.fixture
def feature_params_for_fake_data(
        categorical_feature: List[str],
        numerical_feature: List[str],
) -> FeatureParams:
    params = FeatureParams(
        categorical_features=[categorical_feature],
        numerical_features=[numerical_feature],
        target_feature=None,
    )
    return params


def test_process_categorical_features(
        fake_data: pd.DataFrame,
        feature_params_for_fake_data: FeatureParams,
):
    transformer = build_transformer(feature_params_for_fake_data)
    transformed: pd.DataFrame = transformer.fit_transform(fake_data)
    assert transformed.shape[1] == 4
    assert transformed[:, :3].sum().sum() == 4
    assert np.mean(transformed[:, 3]) < 0.001
    assert np.std(transformed[:, 3]) > 0.99999
    assert np.std(transformed[:, 3]) < 1.0001

