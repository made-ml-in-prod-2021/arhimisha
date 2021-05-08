import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from py._path.local import LocalPath
from sklearn.ensemble import RandomForestRegressor

from src.data.make_data import read_data, get_target_data, get_features_data
from src.entities import ModelParams
from src.entities.feature_params import FeatureParams
from src.features_transformer.build_features_transformer import build_transformer
from src.models.model_fit_predict import (
    train_model,
    predict_model,
    evaluate_model,
    serialize_object
)


@pytest.fixture()
def features_and_target(
        dataset_path: str,
        feature_params: FeatureParams,
) -> Tuple[pd.DataFrame, np.ndarray]:
    data = read_data(dataset_path)
    features = get_features_data(data, feature_params)
    target = get_target_data(data, feature_params).values.ravel()
    transformer = build_transformer(feature_params)
    features = transformer.fit_transform(features)
    return features, target


def test_train_and_predict_model(
        features_and_target: Tuple[pd.DataFrame, np.ndarray],
        model_params: ModelParams
):
    features, target = features_and_target
    model = train_model(features,
                        target,
                        model_params=[model_params],
                        model_name=model_params.model_name)
    assert predict_model(model, features).shape[0] == target.shape[0]


def test_evaluate_model(
        features_and_target: Tuple[pd.DataFrame, np.ndarray],
        model_params: ModelParams
):
    features, target = features_and_target
    model = train_model(features,
                        target,
                        model_params=[model_params],
                        model_name=model_params.model_name)
    predict = predict_model(model, features)
    scores = evaluate_model(target, predict)
    for score in scores.values():
        assert score > 0.8


def test_serialize_model(tmpdir: LocalPath):
    expected_output = tmpdir.join("model.pkl")
    n_estimators = 10
    model = RandomForestRegressor(n_estimators=n_estimators)
    real_output = serialize_object(model, expected_output)
    assert real_output == expected_output
    assert os.path.exists
    with open(real_output, "rb") as f:
        model = pickle.load(f)
    assert isinstance(model, RandomForestRegressor)
