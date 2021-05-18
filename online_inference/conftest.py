import os

import pytest

@pytest.fixture()
def dataset_for_predict_path():
    current_dir = os.path.dirname(__file__)
    return os.path.join(current_dir, "data/test_data_for_predict.csv")
