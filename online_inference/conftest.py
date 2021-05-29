import os
import pytest


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
