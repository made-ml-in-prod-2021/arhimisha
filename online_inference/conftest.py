import os
import pytest
import numpy as np
import pandas as pd
from typing import Union

Array = Union[pd.DataFrame, np.ndarray]


class mocked_model:
    def predict(self, data: Array) -> Array:
        return np.zeros(shape=data.shape[0])


class mocked_transformer:
    def transform(self, data: Array) -> Array:
        return data


@pytest.fixture()
def mocking_model():
    return mocked_model()


@pytest.fixture()
def mocking_transformer():
    return mocked_transformer()


@pytest.fixture()
def dataset_for_predict_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "data/test_data_for_predict.csv")


@pytest.fixture()
def bad_data_1():
    return """{
                    "data": [
                            [
                                0, 1
                            ],
                            [
                                0, 1
                            ]
                        ],
                    "features": {
                        "1": "feature_1",
                        "2": "feature_2"
                    }
                }"""


@pytest.fixture()
def bad_data_2():
    return """{
                    "data": [
                         0, 1                
                    ],
                    "features": [
                        "feature_1",
                        "feature_2"
                    ]
                }"""
