import pickle
from typing import Dict, Union

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

#from ..entities.train_params import TrainingParams




def train_model(
    features: pd.DataFrame, target: pd.Series
) -> LinearSVC:
    model = LinearSVC()
    model.fit(features, target)
    return model


def predict_model(
    model: SklearnRegressionModel, features: pd.DataFrame, use_log_trick: bool = True
) -> np.ndarray:
    predicts = model.predict(features)
    if use_log_trick:
        predicts = np.exp(predicts)
    return predicts


def evaluate_model(
    predicts: np.ndarray, target: pd.Series, use_log_trick: bool = False
) -> Dict[str, float]:
    if use_log_trick:
        target = np.exp(target)
    return {
        "r2_score": r2_score(target, predicts),
        "rmse": mean_squared_error(target, predicts, squared=False),
        "mae": mean_absolute_error(target, predicts),
    }


def serialize_model(model: SklearnRegressionModel, output: str) -> str:
    with open(output, "wb") as f:
        pickle.dump(model, f)
    return output
