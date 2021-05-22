import pandas as pd
import numpy as np
from typing import Tuple, Union
from sklearn.model_selection import train_test_split

from ..entities import SplittingParams
from ..entities import FeatureParams

DataForSplit = Union[pd.DataFrame, np.ndarray]


def read_data(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data


def get_target_data(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    target = df[params.target_feature]
    return target


def get_features_data(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    columns = [column for column in df.columns if (column in params.categorical_features or
                                                   column in params.numerical_features)]
    features = df[columns]
    return features


def split_data(
        features: DataForSplit, target: DataForSplit,
        split_size: float, random_state: int,
        stratify: DataForSplit
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(features, target,
                            test_size=split_size,
                            random_state=random_state,
                            stratify=stratify)


def split_train_valid_data(
        features: DataForSplit, target: DataForSplit,
        params: SplittingParams, stratify: DataForSplit
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return split_data(
        features,
        target,
        params.val_size,
        params.random_state,
        stratify=stratify
    )


def split_train_test_data(
        features: DataForSplit, target: DataForSplit,
        params: SplittingParams, stratify: DataForSplit
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return split_data(
        features,
        target,
        params.test_size,
        params.random_state,
        stratify=stratify
    )
