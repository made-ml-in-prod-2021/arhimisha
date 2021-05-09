import numpy as np
import pandas as pd
from typing import Union
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin

from ..entities.feature_params import FeatureParams

Array_2D = Union[pd.DataFrame, np.ndarray]

class MyStandardScaler(TransformerMixin):
    def transform(self, X: Array_2D, *_) ->Array_2D:
        return (X - self.means) / self.std

    def fit(self, X: Array_2D, *_) ->TransformerMixin:
        self.means = np.mean(np.array(X), axis=0, keepdims=True)
        self.std = np.std(np.array(X), axis=0, keepdims=True)
        self.std[self.std == 0.] = 1
        return self


def build_categorical_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="most_frequent")),
            ("one_hot", OneHotEncoder()),
        ]
    )
    return categorical_pipeline


def build_numerical_pipeline() -> Pipeline:
    num_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ('scaler', MyStandardScaler())
        ]
    )
    return num_pipeline


def build_transformer(params: FeatureParams) -> ColumnTransformer:
    transformer = ColumnTransformer(
        [
            (
                "categorical_pipeline",
                build_categorical_pipeline(),
                params.categorical_features,
            ),
            (
                "numerical_pipeline",
                build_numerical_pipeline(),
                params.numerical_features,
            ),
        ]
    )
    return transformer
