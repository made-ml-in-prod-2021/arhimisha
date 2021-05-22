from src.data.make_data import (
    read_data,
    get_target_data,
    get_features_data,
    split_train_valid_data,
    split_train_test_data
)
from src.entities import SplittingParams, FeatureParams


def test_load_dataset(dataset_path: str):
    data = read_data(dataset_path)
    assert data.shape[0] >= 100
    assert data.shape[1] == 14


def test_get_feature_data(dataset_path: str, feature_params: FeatureParams):
    data = read_data(dataset_path)
    for feature in feature_params.numerical_features:
        assert feature in data.keys()
    for feature in feature_params.categorical_features:
        assert feature in data.keys()
    features_data = get_features_data(data, feature_params)
    assert features_data.shape == (len(data),
                                   len(feature_params.numerical_features) +
                                   len(feature_params.categorical_features))


def test_get_target_data(dataset_path: str, feature_params: FeatureParams):
    data = read_data(dataset_path)
    assert feature_params.target_feature[0] in data.keys()
    target_data = get_target_data(data, feature_params)
    assert target_data.shape == (len(data), 1)


def test_split_train_valid_data(
        dataset_path: str,
        feature_params: FeatureParams,
        splitting_params: SplittingParams
):
    data = read_data(dataset_path)
    features_data = get_features_data(data, feature_params)
    target_data = get_target_data(data, feature_params)

    X_train, X_valid, y_train, y_valid = split_train_valid_data(
        features_data,
        target_data,
        splitting_params,
        target_data
    )
    check_asserts(X_train, X_valid, y_train, y_valid)


def test_split_train_test_data(
        dataset_path: str,
        feature_params: FeatureParams,
        splitting_params: SplittingParams
):
    data = read_data(dataset_path)
    features_data = get_features_data(data, feature_params).values
    target_data = get_target_data(data, feature_params).values

    X_train, X_test, y_train, y_test = split_train_test_data(
        features_data,
        target_data,
        splitting_params,
        target_data
    )
    check_asserts(X_train, X_test, y_train, y_test)


def check_asserts(X_train, X_test, y_train, y_test):
    assert X_train.shape[0] >= 90
    assert X_test.shape[0] >= 9
    assert y_train.shape[0] >= 90
    assert y_test.shape[0] >= 9
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
    assert X_train.shape[1] == X_test.shape[1]
    assert y_train.shape[1] == y_test.shape[1]
