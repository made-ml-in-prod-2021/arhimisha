import pickle
from typing import Dict, Union, List, Any

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

from ..entities.model_params import ModelParams

SklearnRegressionModel = Union[LinearSVC]


def train_model(
        features: np.ndarray, target: np.ndarray,
        model_params: List[ModelParams], model_name
) -> LinearSVC:
    params = [item for item in model_params if item.model_name == model_name]
    if model_name == "LinearSVC":
        model = LinearSVC(**(params[0].params))
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def predict_model(
        model: SklearnRegressionModel, features: np.ndarray
) -> np.ndarray:
    return model.predict(features)


def evaluate_model(
        target: np.ndarray, predicts: np.ndarray
) -> Dict[str, float]:
    return {
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }


def serialize_object(obj: Any, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(obj, f)
    return output
