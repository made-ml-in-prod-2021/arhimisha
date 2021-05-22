import json
import logging
import sys

import click

from src.data import (
    read_data,
    get_target_data,
    get_features_data,
    split_train_valid_data,
    split_train_test_data
)
from src.entities.train_pipeline_params import (
    TrainingPipelineParams,
    read_training_pipeline_params,
)
from src.features_transformer.build_features_transformer import build_transformer
from src.models.model_fit_predict import (
    train_model,
    serialize_object,
    predict_model,
    evaluate_model,
)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def train_pipeline(training_pipeline_params: TrainingPipelineParams, model_name: str):
    logger.info(f"start train pipeline with params {training_pipeline_params}")
    data = read_data(training_pipeline_params.input_data_path)
    logger.info(f"data.shape is {data.shape}")

    feature_data = get_features_data(data, training_pipeline_params.feature_params)
    target_data = get_target_data(data, training_pipeline_params.feature_params)
    target_data = target_data.values.ravel()

    logger.info(f"feature_data.shape is {feature_data.shape}")
    logger.info(f"target_data.shape is {target_data.shape}")

    X_data, X_valid, y_data, y_valid = split_train_valid_data(
        feature_data,
        target_data,
        training_pipeline_params.splitting_params,
        stratify=target_data
    )
    logger.info(f"X_data.shape is {X_data.shape}")
    logger.info(f"y_data.shape is {y_data.shape}")
    logger.info(f"X_valid.shape is {X_valid.shape}")
    logger.info(f"y_valid.shape is {y_valid.shape}")

    transformer = build_transformer(training_pipeline_params.feature_params)
    X_data = transformer.fit_transform(X_data)
    X_valid = transformer.transform(X_valid)

    X_train, X_test, y_train, y_test = split_train_test_data(
        X_data,
        y_data,
        training_pipeline_params.splitting_params,
        stratify=y_data
    )

    logger.info(f"X_train.shape is {X_train.shape}")
    logger.info(f"y_train.shape is {y_train.shape}")
    logger.info(f"X_test.shape is {X_test.shape}")
    logger.info(f"y_test.shape is {y_test.shape}")

    model = train_model(
        X_train, y_train, training_pipeline_params.models_params, model_name
    )

    logger.info(f"model is {model}")

    predicts_test = predict_model(model, X_test)
    predicts_valid = predict_model(model, X_valid)

    metrics_test = evaluate_model(y_test, predicts_test)
    metrics_valid = evaluate_model(y_valid, predicts_valid)

    with open(training_pipeline_params.metrics_path, "w") as metric_file:
        json.dump([metrics_test, metrics_valid], metric_file)
    logger.info(f"metrics_test is {metrics_test}")
    logger.info(f"metrics_valid is {metrics_valid}")

    path_to_model = serialize_object(model, training_pipeline_params.output_model_path)
    path_to_transformer = serialize_object(transformer, training_pipeline_params.output_transformer_path)

    return path_to_model, path_to_transformer, metrics_valid


@click.command(name="train_pipeline")
@click.argument("config_path")
@click.argument("model_name")
def train_pipeline_command(config_path: str, model_name: str):
    params = read_training_pipeline_params(config_path)
    train_pipeline(params, model_name)


if __name__ == "__main__":
    train_pipeline_command()
